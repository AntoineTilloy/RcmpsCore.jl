"""Lindblad action without forming a D² by D² transfer matrix."""
lindblad(q::AbstractMatrix,r::AbstractMatrix,x::AbstractMatrix) = q*x+x*q'+r*x*r'
lindblad(s::State,x::AbstractMatrix) = lindblad(s.Q,s.R,x)
lindblad_adjoint(s::State,x::AbstractMatrix) = s.Q'*x+x*s.Q+s.R'*x*s.R

function lindblad!(out,s::State,x,tmp=similar(x))
    mul!(out,s.Q,x)
    mul!(out,x,s.Q',1,1)
    mul!(tmp,s.R,x)
    mul!(out,tmp,s.R',1,1)
    return out
end
function lindblad_adjoint!(out,s::State,x,tmp=similar(x))
    mul!(out,s.Q',x)
    mul!(out,x,s.Q,1,1)
    mul!(tmp,s.R',x)
    mul!(out,tmp,s.R,1,1)
    return out
end

"""Dense reference using Julia column-major vec: vec(L(X)) == generate_T(s)*vec(X)."""
function transfer_matrix(q,r)
    id = Matrix{ComplexF64}(I,size(q)...)
    return kron(id,q)+kron(conj(q),id)+kron(conj(r),r)
end
generate_T(s::State) = transfer_matrix(s.Q,s.R)

struct SolverFailure <: Exception
    message::String
end
Base.showerror(io::IO,e::SolverFailure) = print(io,e.message)

function checked_density(q,r,rho; tol=1e-11)
    size(rho) == size(q) || throw(DimensionMismatch("rho has the wrong dimensions"))
    all(isfinite,rho) || throw(SolverFailure("nonfinite stationary state"))
    z = tr(rho)
    abs(z) > eps()*norm(rho) || throw(SolverFailure("stationary state has zero trace"))
    # Remove the arbitrary complex phase BEFORE hermitizing a Krylov eigenvector.
    x = Matrix{ComplexF64}(rho/z)
    norm(x-x') <= 100tol || throw(SolverFailure("stationary state is not Hermitian"))
    x = (x+x')/2
    residual = norm(lindblad(q,r,x))
    scale = max(1.0,2opnorm(q)+opnorm(r)^2)
    residual <= tol*scale || throw(SolverFailure("stationary residual $residual exceeds $(tol*scale)"))
    eigmin(Hermitian(x)) >= -10tol || throw(SolverFailure("stationary state is not positive"))
    return x # Do not clip eigenvalues or shift rho: that changes the stationary equation.
end

"""Cached Schur factorization of the no-jump part (robust for defective Q)."""
struct NoJumpInverse
    T::Matrix{ComplexF64}
    U::Matrix{ComplexF64}
    shift::Float64
end
function NoJumpInverse(q,r)
    f = schur(q)
    scale = max(1.0,opnorm(q),opnorm(r)^2)
    # A dummy jump sqrt(shift)I makes S invertible even for a dark state.
    shift = maximum(real,f.values) >= -100eps()*scale ? scale : 0.0
    t = Matrix(f.T) - (shift/2)*I
    return NoJumpInverse(t,Matrix(f.Z),shift)
end
function (p::NoJumpInverse)(y; adjoint=false)
    c = p.U'*y*p.U
    _, scale = LAPACK.trsyl!(adjoint ? 'C' : 'N',adjoint ? 'N' : 'C',p.T,p.T,c)
    c .*= -1/scale
    return p.U*c*p.U'
end

"""Find rho via the CPTP map Phi(X)=R*(-S)^(-1)(X)*R', reconstructing
rho proportional to (-S)^(-1)(xi), as in arXiv:2608.30860. Returns diagnostics.
`:dense` is an independent small-system reference; `:auto` uses CPTP Arnoldi.
"""
function steady_state(q::AbstractMatrix,r::AbstractMatrix; method=:auto,tol=1e-11,
                      krylovdim=40,maxiter=200)
    tol > 0 && maxiter > 0 && krylovdim > 1 || throw(ArgumentError("invalid solver controls"))
    d = size(q,1)
    id = Matrix{ComplexF64}(I,d,d)
    if d == 1
        return (rho=id,residual=0.0,method=:scalar,iterations=0)
    end
    if d <= 8 && method != :dense
        l = transfer_matrix(q,r)
        count(<(tol*max(1,opnorm(l))),svdvals(l)) == 1 ||
            throw(SolverFailure("stationary state is nonunique or numerically unresolved"))
    end
    if method == :dense
        d <= 64 || throw(ArgumentError("dense reference is limited to D≤64"))
        l = transfer_matrix(q,r)
        # This also rejects a nonunique stationary state rather than selecting one silently.
        count(<(tol*max(1,opnorm(l))),svdvals(l)) == 1 ||
            throw(SolverFailure("stationary state is nonunique or numerically unresolved"))
        v = vec(id)
        rho = reshape((l+v*v'/d)\(v/d),d,d)
        iterations = 1
    elseif method in (:auto,:cptp)
        p = NoJumpInverse(q,r)
        phi(x) = begin
            y = p(x)
            r*y*r' + p.shift*y
        end
        # Largest real part selects +1 even when Phi has other peripheral eigenvalues.
        vals,vecs,info = eigsolve(phi,id/d,1,:LR;tol=tol/10,
                                krylovdim=min(d*d,krylovdim),maxiter=maxiter)
        info.converged >= 1 && abs(vals[1]-1) <= 100tol ||
            throw(SolverFailure("CPTP Arnoldi did not converge to eigenvalue 1"))
        rho = p(vecs[1])
        iterations = info.numops
    else
        throw(ArgumentError("unknown stationary solver: $method"))
    end
    rho = checked_density(q,r,rho;tol)
    return (rho=rho,residual=norm(lindblad(q,r,rho)),method=method==:auto ? :cptp : method,
            iterations=iterations)
end
steady_state(s::State;kw...) = steady_state(s.Q,s.R;kw...)
get_ρ0(s::State;kw...) = steady_state(s;kw...).rho

"""Solve (-L* + |I><rho|)H = A - tr(rho A)I, fixing tr(rho H)=0.
This is the exact initial-state pullback for a finite-window observable.
"""
function boundary_environment(s::State,a;tol=1e-11,krylovdim=40,maxiter=200,method=:iterative)
    id = Matrix{ComplexF64}(I,s.D,s.D)
    b = a-tr(s.ρ0*a)*id
    s.D == 1 && return zero(a)
    op(h) = -lindblad_adjoint(s,h)+tr(s.ρ0*h)*id
    if method == :dense
        s.D <= 64 || throw(ArgumentError("dense reference is limited to D≤64"))
        h = reshape((-generate_T(s)'+vec(id)*vec(s.ρ0)')\vec(b),s.D,s.D)
    elseif method == :iterative
        p = NoJumpInverse(s.Q,s.R)
        # Sherman–Morrison incorporates |I><rho| into the no-jump inverse.
        z = p(id;adjoint=true)
        den = 1+tr(s.ρ0*z)
        pre(y) = begin
            x = p(y;adjoint=true)
            x - (tr(s.ρ0*x)/den)*z
        end
        y,info = linsolve(x->op(pre(x)),b;tol=tol*max(1,norm(b)),
                         krylovdim=min(s.D^2,krylovdim),maxiter=maxiter)
        h = pre(y)
        if info.converged < 1 || norm(op(h)-b) > 20tol*max(1,norm(b))
            s.D <= 16 || throw(SolverFailure("boundary Poisson solve failed"))
            # A bounded small-system fallback avoids cancellation in an ill-conditioned
            # right preconditioner. It must pass the same physical residual check.
            h = reshape((-generate_T(s)'+vec(id)*vec(s.ρ0)')\vec(b),s.D,s.D)
        end
    else
        throw(ArgumentError("unknown boundary solver: $method"))
    end
    norm(op(h)-b) <= 20tol*max(1,norm(b)) || throw(SolverFailure("boundary residual too large"))
    return h
end

function entanglement_entropy(s::State;alpha=1.0)
    alpha > 0 || throw(ArgumentError("Renyi order must be positive"))
    p = max.(eigvals(Hermitian(s.ρ0)),0)
    p ./= sum(p)
    return alpha ≈ 1 ? -sum(x->iszero(x) ? 0.0 : x*log(x),p) : log(sum(p.^alpha))/(1-alpha)
end
"""Inverse auxiliary correlation length; this is not a Hamiltonian excitation energy."""
function transfer_gap(s::State;method=:auto,krylovdim=60,maxiter=300,tol=1e-10)
    s.D == 1 && return Inf
    if method == :dense || (method == :auto && s.D <= 16)
        return sort(-real.(eigvals(generate_T(s))))[2]
    end
    method in (:auto,:iterative) || throw(ArgumentError("unknown gap method"))
    # A random start is essential: rho or I may span only the zero mode.
    initial = randn(MersenneTwister(581),ComplexF64,s.D,s.D)
    vals,vs,info = eigsolve(x->lindblad(s,x),initial,2,:LR;
                           tol,krylovdim=min(s.D^2,krylovdim),maxiter)
    info.converged >= 2 || throw(SolverFailure("gap eigensolver did not converge"))
    return sort(-real.(vals))[2]
end
"""Legacy name for `transfer_gap(s)`. For a Hamiltonian rest energy use
`excitation_gap(s)` (or `mass_gap(s, model)` with an explicit ExcitationModel).
"""
mass_gap(s::State;kw...) = transfer_gap(s;kw...)
robust_inv(rho,reg) = inv(Hermitian(rho+reg*I))
robust_inv(rho,m,reg) = Hermitian(rho+reg*I)\m
