"""Controls for the matrix-free kinetic resolvent backend.
`tolerance` targets the true relative residual of every forward/adjoint shifted
solve. At basis exhaustion or `maxdim`, `fallback_tolerance` permits a larger
explicitly checked residual. It defaults to `tolerance` (strict acceptance).
`maxdim` caps each Arnoldi basis; angular accuracy is separately controlled
by `KineticOptions`. The physical transfer matrix is never formed.
"""
Base.@kwdef struct KineticKrylovOptions
    tolerance::Float64 = 1e-12
    fallback_tolerance::Float64 = tolerance
    initialdim::Int = 16
    maxdim::Int = 512
end

# Scratch space belongs to one evaluation; the State is only read. Four bases
# at most are retained (one forward and adjoint RHS per kinetic contribution).
# Storage grows with the actual basis size, bounded by maxdim, and angular
# solutions are bounded separately by KineticOptions.maxnodes.
mutable struct KineticArnoldiBasis
    V::Matrix{ComplexF64}
    H::Matrix{ComplexF64}
    rhs::Vector{ComplexF64}
    beta::Float64
    k::Int
    exhausted::Bool
end
function KineticArnoldiBasis(rhs,capacity)
    b=vec(copy(rhs)); beta=norm(b)
    isfinite(beta) || throw(SolverFailure("nonfinite kinetic right-hand side"))
    V=zeros(ComplexF64,length(b),capacity+1)
    iszero(beta) || (V[:,1].=b./beta)
    KineticArnoldiBasis(V,zeros(ComplexF64,capacity+1,capacity),b,beta,0,iszero(beta))
end
mutable struct KineticKrylovContext
    state::State
    tolerance::Float64
    fallback_tolerance::Float64
    initialdim::Int
    maxdim::Int
    bases::Dict{Tuple{Bool,Bool},KineticArnoldiBasis}
    actions::Int
    checked_solves::Int
    fallback_solves::Int
    maximum_relative_residual::Float64
end
function KineticKrylovContext(s, p::KineticKrylovOptions)
    isfinite(p.tolerance) && p.tolerance>0 &&
        isfinite(p.fallback_tolerance) && p.fallback_tolerance>=p.tolerance &&
        p.initialdim>=1 && p.maxdim>=p.initialdim ||
        throw(ArgumentError("invalid kinetic Krylov controls"))
    KineticKrylovContext(s,p.tolerance,p.fallback_tolerance,min(p.initialdim,s.D^2),min(p.maxdim,s.D^2),
                        Dict{Tuple{Bool,Bool},KineticArnoldiBasis}(),0,0,0,0.0)
end
kinetic_context(s,::Val{:krylov},p::KineticKrylovOptions)=KineticKrylovContext(s,p)

function kinetic_action!(out,ctx,x,adj,tmp)
    adj ? lindblad_adjoint!(out,ctx.state,x,tmp) : lindblad!(out,ctx.state,x,tmp)
    ctx.actions+=1
    out
end
function kinetic_extend!(basis,ctx,adj,target)
    d=ctx.state.D
    if size(basis.H,2)<target
        V=zeros(ComplexF64,d*d,target+1)
        H=zeros(ComplexF64,target+1,target)
        V[:,1:basis.k+1].=view(basis.V,:,1:basis.k+1)
        H[1:basis.k+1,1:basis.k].=view(basis.H,1:basis.k+1,1:basis.k)
        basis.V,basis.H=V,H
    end
    out,tmp=zero(ctx.state.R),zero(ctx.state.R)
    for j in basis.k+1:target
        kinetic_action!(out,ctx,reshape(view(basis.V,:,j),d,d),adj,tmp)
        v=vec(out)
        all(isfinite,v) || throw(SolverFailure("nonfinite kinetic Arnoldi action"))
        oldnorm=norm(v)
        # Twice-reorthogonalized modified projection; retain H from both passes.
        block=view(basis.V,:,1:j)
        for _ in 1:2
            h=block'*v
            view(basis.H,1:j,j).+=h
            v .-=block*h
        end
        beta=norm(v)
        basis.H[j+1,j]=beta
        basis.k=j
        if beta <= 10eps(Float64)*max(oldnorm,eps(Float64)) || j==d*d
            basis.exhausted=true
            break
        end
        basis.V[:,j+1].=v./beta
    end
end
# Ritz values of a nonnormal operator can give an unusable projected shift
# before the basis converges. Enlarge the basis rather than accept that solve.
function kinetic_projected_solutions(basis,nodes)
    k,n=basis.k,length(nodes)
    try
        f=schur(Matrix(view(basis.H,1:k,1:k)))
        reduced_rhs=basis.beta*conj.(f.Z[1,:])
        Y=Matrix{ComplexF64}(undef,k,n)
        estimate=0.0
        for (i,c) in enumerate(nodes)
            Y[:,i]=f.Z*(UpperTriangular(I-c*f.T)\reduced_rhs)
            estimate=max(estimate,abs(c*basis.H[k+1,k]*Y[k,i])/basis.beta)
        end
        all(isfinite,Y) || return nothing,Inf
        return Y,estimate
    catch error
        error isa LinearAlgebra.LAPACKException || error isa LinearAlgebra.SingularException || rethrow()
        return nothing,Inf
    end
end
function kinetic_solutions(ctx,rhs,comm,adj,nodes)
    basis=get!(ctx.bases,(comm,adj)) do
        KineticArnoldiBasis(rhs,ctx.initialdim)
    end
    # Cached bases may only be reused for the same right-hand side.
    basis.rhs==vec(rhs) || throw(SolverFailure("Krylov context was reused after state mutation"))
    N,n=length(basis.rhs),length(nodes)
    iszero(basis.beta) && return zeros(ComplexF64,N,n)
    last_true_residual=Inf
    while true
        if iszero(basis.k)
            kinetic_extend!(basis,ctx,adj,ctx.initialdim)
        end
        k=basis.k
        Y,estimate=kinetic_projected_solutions(basis,nodes)
        if !isnothing(Y) && (estimate <= ctx.tolerance || basis.exhausted || k==ctx.maxdim)
            X=view(basis.V,:,1:k)*Y
            d=ctx.state.D
            out,tmp=zero(ctx.state.R),zero(ctx.state.R)
            worst=0.0
            for (i,c) in enumerate(nodes)
                x=reshape(view(X,:,i),d,d)
                kinetic_action!(out,ctx,x,adj,tmp)
                residual=norm(view(X,:,i)-c*vec(out)-basis.rhs)/basis.beta
                worst=max(worst,residual)
            end
            last_true_residual=worst
            terminal=basis.exhausted || k==ctx.maxdim
            fallback=terminal && worst>ctx.tolerance && worst<=ctx.fallback_tolerance
            if isfinite(worst) && (worst<=ctx.tolerance || fallback)
                ctx.checked_solves+=n
                ctx.maximum_relative_residual=max(ctx.maximum_relative_residual,worst)
                if fallback
                    ctx.fallback_solves+=n
                    @warn "Kinetic Krylov accepted checked residual at fallback tolerance" basis_size=k true_relative_residual=worst target=ctx.tolerance ceiling=ctx.fallback_tolerance adjoint=adj commutator=comm
                end
                return X
            end
            basis.exhausted && throw(SolverFailure("Arnoldi exhausted at $k vectors with true relative residual $worst"))
        end
        (basis.exhausted || basis.k==ctx.maxdim) && throw(SolverFailure("shifted Arnoldi failed at $(ctx.maxdim) vectors; projected residual $estimate; last true residual $last_true_residual"))
        kinetic_extend!(basis,ctx,adj,min(2basis.k,ctx.maxdim))
    end
end
function kinetic_components(s,spec,ctx::KineticKrylovContext,grid;gradient=true)
    ctx.state===s || throw(SolverFailure("Krylov context belongs to another state"))
    commutator=spec.commutator
    op=commutator ? s.Q*s.R-s.R*s.Q : s.R
    theta,weights=grid
    nodes=cos.(theta)
    xs=kinetic_solutions(ctx,s.ρ0*op',commutator,false,nodes)
    value=sum(w*real(dot(vec(op'),view(xs,:,i))) for (i,w) in enumerate(weights))
    isfinite(value) || throw(SolverFailure("nonfinite kinetic value"))
    gradient || return value
    # Hilbert-Schmidt adjoint solve, then adjoint its matrix to obtain trace dual.
    aas=kinetic_solutions(ctx,Matrix(op'),commutator,true,nodes)
    m,nn,u,v,env=(zero(s.R) for _ in 1:5)
    for (i,w) in enumerate(weights)
        z=nodes[i]
        x=reshape(view(xs,:,i),s.D,s.D)
        a=reshape(view(aas,:,i),s.D,s.D)'
        m .+= w*z .* (x*s.R'*a-x*a*s.R')
        nn .+= w*z .* (a*s.R*x-s.R*a*x)
        u .+= w .* x
        v .+= w .* (a*s.ρ0)
        env .+= w .* (op'*a)
    end
    if commutator
        m .+= u*s.Q-s.Q*u-s.R*u*s.R'+u*s.R*s.R'
        nn .+= s.Q'*v-v*s.Q'-s.R*v*s.R'+s.R*s.R'*v
    else
        m .+= u; nn .+=v
    end
    covector=(m+nn')/2;env=(env+env')/2
    all(isfinite,covector) && all(isfinite,env) || throw(SolverFailure("nonfinite kinetic gradient components"))
    return value,covector,env
end
