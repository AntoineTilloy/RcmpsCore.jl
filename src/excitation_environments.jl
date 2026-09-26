# Four tangent channels, ordered (00, ket, bra, ket+bra). See docs/excitations.tex.
# The state is snapshotted by ExcitationContext: scratch and factors are evaluation-local.
struct ExcitationContext{F,S}
    state::State
    momentum::Float64
    transfer::Matrix{ComplexF64}
    poisson::F
    ketfactor::F
    brafactor::F
    schur::S
    tol::Float64
end

"""Dense transfer factors for tangent environments at momentum `p`, on a snapshot of `s`.
Retained storage is about 48D⁴ bytes (Schur T, Z and one LU) at p=0 and 80D⁴ otherwise,
plus 16D⁴ with `store_transfer=true` (the generator itself, used by research scripts).
"""
function ExcitationContext(s::State,p::Real;tol=1e-10,max_bond=16,store_transfer=true)
    isfinite(p) && isfinite(tol) && tol > 0 || throw(ArgumentError("invalid momentum or environment tolerance"))
    s.D <= max_bond || throw(ArgumentError("dense excitation reference is limited to D≤$max_bond; increase max_bond explicitly to accept its cost"))
    t = deepcopy(s)
    l = generate_T(t)
    f = schur(l)
    # Reject peripheral modes too: infinite-line boundary integrals need mixing.
    # The Schur diagonal gives the spectrum; the Frobenius norm bounds the spectral norm
    # without an additional O(D⁶) factorization.
    values = f.values
    scale = max(1,norm(l))
    count(z->abs(z)<=tol*scale,values) == 1 &&
        all(z->abs(z)<=tol*scale || real(z)<-tol*scale,values) ||
        throw(SolverFailure("excitation environments require a unique, mixing transfer fixed point"))
    poisson,ket,bra = transfer_factors(t,l,p)
    return ExcitationContext(t,Float64(p),store_transfer ? l : Matrix{ComplexF64}(undef,0,0),
                             poisson,ket,bra,f,Float64(tol))
end

# Pinned factors of -L, -L+ip and -L-ip; at p=0 all three coincide.
function transfer_factors(t,l,p;poisson=nothing)
    projector = vec(t.ρ0)*vec(Matrix{ComplexF64}(I,t.D,t.D))'
    isnothing(poisson) && (poisson = lu(-l+projector))
    iszero(p) && return poisson,poisson,poisson
    return poisson,lu(-l+im*p*I+projector),lu(-l-im*p*I+projector)
end

# The same state with independent (deep-copied) settings; matrices are shared read-only.
share_state(s::State;integrator=s.integrator_param) =
    State(s.D,s.R,s.K,s.Q,s.ρ0,deepcopy(s.model_param),deepcopy(integrator),
          deepcopy(s.manifold_param),s.e0,s.grad,s.info)

"""
    ExcitationContext(base; momentum=base.momentum, integrator=base.state.integrator_param)

Reuse the momentum-independent factors of `base` (Schur form and pinned Poisson LU)
for another momentum or other integration settings. Only the two momentum-shifted
LU factors are recomputed, and only when the momentum changes to a nonzero value.
The returned context has its own copy of the settings; matrices are shared read-only.
"""
function ExcitationContext(base::ExcitationContext;momentum::Real=base.momentum,
                           integrator::IntegratorParameters=base.state.integrator_param)
    isfinite(momentum) || throw(ArgumentError("invalid momentum"))
    t = share_state(base.state;integrator)
    p = Float64(momentum)
    ket,bra = if p == base.momentum
        base.ketfactor,base.brafactor
    else
        l = isempty(base.transfer) ? generate_T(t) : base.transfer
        _,k,b = transfer_factors(t,l,p;poisson=base.poisson)
        k,b
    end
    return ExcitationContext(t,p,base.transfer,base.poisson,ket,bra,base.schur,base.tol)
end

function excitation_poisson(ctx,b;shift=0)
    s = ctx.state
    rhs = b-tr(b)*s.ρ0
    f = shift == 0 ? ctx.poisson : shift == 1 ? ctx.ketfactor : ctx.brafactor
    x = reshape(f\vec(rhs),s.D,s.D)
    x -= tr(x)*s.ρ0
    residual = -lindblad(s,x)+(im*shift*ctx.momentum)*x-rhs
    norm(residual) <= 20ctx.tol*max(1,norm(rhs)) || throw(SolverFailure("tangent boundary solve residual too large"))
    return x
end

"""The physical complex tangent inner product, with no optimizer regularization."""
function tangent_metric(s::State,u::AbstractMatrix,w::AbstractMatrix)
    size(u) == size(w) == size(s.R) || throw(DimensionMismatch("tangent dimensions do not match the state"))
    return tr(w*s.ρ0*u')
end

struct ExcitationPair
    context::ExcitationContext
    w::Matrix{ComplexF64}
    u::Matrix{ComplexF64}
    v::Matrix{ComplexF64}
    vbra::Matrix{ComplexF64}
    overlap::ComplexF64
    right::Array{ComplexF64,3}
    left00::Matrix{ComplexF64}
end
ket_insertion(s,v,w,x) = v*x+w*x*s.R'
bra_insertion(s,v,u,x) = x*v'+s.R*x*u'

function ExcitationPair(ctx::ExcitationContext,u::AbstractMatrix,w::AbstractMatrix)
    s = ctx.state
    size(u) == size(w) == size(s.R) || throw(DimensionMismatch("tangent dimensions do not match the state"))
    all(isfinite,u) && all(isfinite,w) || throw(ArgumentError("nonfinite tangent"))
    u,w = Matrix{ComplexF64}(u),Matrix{ComplexF64}(w)
    v,vbra = -s.R'*w,-s.R'*u
    n = tangent_metric(s,u,w)
    right = zeros(ComplexF64,s.D,s.D,4)
    right[:,:,1] = s.ρ0
    right[:,:,2] = excitation_poisson(ctx,ket_insertion(s,v,w,s.ρ0);shift=1)
    right[:,:,3] = excitation_poisson(ctx,bra_insertion(s,vbra,u,s.ρ0);shift=-1)
    f = ket_insertion(s,v,w,right[:,:,3])+bra_insertion(s,vbra,u,right[:,:,2])+w*s.ρ0*u'
    right[:,:,4] = excitation_poisson(ctx,f)
    # Use the same adjoint Poisson equation as boundary_environment, reusing its factor.
    b = u'*w-n*I
    h = reshape(ctx.poisson'\vec(b),s.D,s.D)
    h -= tr(s.ρ0*h)*I
    norm(-lindblad_adjoint(s,h)-b) <= 20ctx.tol*max(1,norm(b)) ||
        throw(SolverFailure("tangent right-tail solve residual too large"))
    return ExcitationPair(ctx,w,u,v,vbra,n,right,h)
end

"""Compute the momentum-shifted single-insertion, connected double-insertion,
and adjoint tail environments for <Phi_p(u)|O|Phi_p(w)>. Dense small-D reference.
"""
function excitation_environments(s::State,p::Real,u::AbstractMatrix,w::AbstractMatrix;kw...)
    pair = ExcitationPair(ExcitationContext(s,p;kw...),u,w)
    return (ket=copy(pair.right[:,:,2]),bra=copy(pair.right[:,:,3]),
            double=copy(pair.right[:,:,4]),tail=copy(pair.left00),overlap=pair.overlap)
end

# Matrix-free in each D×D block; no 4D² transfer matrix is formed.
function excitation_transfer!(out,x,pair,tmp)
    s = pair.context.state
    @views for k in 1:4
        lindblad!(out[:,:,k],s,x[:,:,k],tmp)
    end
    @views begin
        out[:,:,2] .-= im*pair.context.momentum .* x[:,:,2]
        out[:,:,3] .+= im*pair.context.momentum .* x[:,:,3]
        for (src,dst) in ((1,2),(3,4))
            mul!(out[:,:,dst],pair.v,x[:,:,src],1,1)
            addtriple!(out[:,:,dst],pair.w,x[:,:,src],s.R',1,tmp)
        end
        for (src,dst) in ((1,3),(2,4))
            mul!(out[:,:,dst],x[:,:,src],pair.vbra',1,1)
            addtriple!(out[:,:,dst],s.R,x[:,:,src],pair.u',1,tmp)
        end
        addtriple!(out[:,:,4],pair.w,x[:,:,1],pair.u',1,tmp)
        out[:,:,4] .-= pair.overlap .* x[:,:,1]
    end
    return out
end

excitation_contract(pair,x) = tr(@view x[:,:,4])+tr(pair.left00*(@view x[:,:,1]))

function excitation_operators(pair,commutator)
    s,p = pair.context.state,pair.context.momentum
    if commutator
        c = s.Q*s.R-s.R*s.Q
        cw = s.Q*pair.w-pair.w*s.Q+pair.v*s.R-s.R*pair.v-im*p*pair.w
        cu = s.Q*pair.u-pair.u*s.Q+pair.vbra*s.R-s.R*pair.vbra-im*p*pair.u
        return c,cw,cu
    end
    return s.R,pair.w,pair.u
end

# Lift a normal-ordered annihilation (left) or creation (right) source.
function excitation_source_add!(out,x,ops,left,a)
    c,cw,cu = ops
    @views for k in 1:4
        if left
            mul!(out[:,:,k],c,x[:,:,k],a,1)
        else
            mul!(out[:,:,k],x[:,:,k],c',a,1)
        end
    end
    @views if left
        mul!(out[:,:,2],cw,x[:,:,1],a,1)
        mul!(out[:,:,4],cw,x[:,:,3],a,1)
    else
        mul!(out[:,:,3],x[:,:,1],cu',a,1)
        mul!(out[:,:,4],x[:,:,2],cu',a,1)
    end
    return out
end

# Triangular Schur form of (1+i z shift p)I - zT, formed once per quadrature node.
function resolvent_factor(ctx,z,shift)
    mat = -z .* ctx.schur.T
    d = 1+im*z*shift*ctx.momentum
    @inbounds for i in axes(mat,1)
        mat[i,i] += d
    end
    return UpperTriangular(mat)
end
function resolvent_factors(ctx,z)
    f0 = resolvent_factor(ctx,z,0)
    iszero(ctx.momentum) && return (f0,f0,f0)
    return (f0,resolvent_factor(ctx,z,1),resolvent_factor(ctx,z,-1))
end

function excitation_resolvent(pair,b,z;factors=resolvent_factors(pair.context,z))
    ctx,s = pair.context,pair.context.state
    f = ctx.schur
    f0,fket,fbra = factors
    solveblock(rhs,factor) = reshape(f.Z*(factor\(f.Z'*vec(rhs))),s.D,s.D)
    x = similar(b)
    x[:,:,1] = solveblock(b[:,:,1],f0)
    x[:,:,2] = solveblock(b[:,:,2]+z*ket_insertion(s,pair.v,pair.w,x[:,:,1]),fket)
    x[:,:,3] = solveblock(b[:,:,3]+z*bra_insertion(s,pair.vbra,pair.u,x[:,:,1]),fbra)
    rhs = b[:,:,4]+z*(ket_insertion(s,pair.v,pair.w,x[:,:,3])+
          bra_insertion(s,pair.vbra,pair.u,x[:,:,2])+pair.w*x[:,:,1]*pair.u'-pair.overlap*x[:,:,1])
    x[:,:,4] = solveblock(rhs,f0)
    return x
end
