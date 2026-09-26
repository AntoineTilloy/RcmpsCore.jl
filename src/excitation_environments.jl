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

function ExcitationContext(s::State,p::Real;tol=1e-10,max_bond=16)
    isfinite(p) && isfinite(tol) && tol > 0 || throw(ArgumentError("invalid momentum or environment tolerance"))
    s.D <= max_bond || throw(ArgumentError("dense excitation reference is limited to D≤$max_bond; increase max_bond explicitly to accept its cost"))
    t = deepcopy(s)
    l = generate_T(t)
    id = Matrix{ComplexF64}(I,t.D,t.D)
    projector = vec(t.ρ0)*vec(id)'
    # Reject peripheral modes too: infinite-line boundary integrals need mixing.
    values = eigvals(l)
    scale = max(1,opnorm(l))
    count(z->abs(z)<=tol*scale,values) == 1 &&
        all(z->abs(z)<=tol*scale || real(z)<-tol*scale,values) ||
        throw(SolverFailure("excitation environments require a unique, mixing transfer fixed point"))
    f = lu(-l+projector)
    return ExcitationContext(t,Float64(p),l,f,lu(-l+im*p*I+projector),
                             lu(-l-im*p*I+projector),schur(l),Float64(tol))
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

function excitation_resolvent(pair,b,z)
    ctx,s = pair.context,pair.context.state
    f = ctx.schur
    function solveblock(rhs,shift)
        factor = UpperTriangular((1+im*z*shift*ctx.momentum)*I-z*f.T)
        reshape(f.Z*(factor\(f.Z'*vec(rhs))),s.D,s.D)
    end
    x = similar(b)
    x[:,:,1] = solveblock(b[:,:,1],0)
    x[:,:,2] = solveblock(b[:,:,2]+z*ket_insertion(s,pair.v,pair.w,x[:,:,1]),1)
    x[:,:,3] = solveblock(b[:,:,3]+z*bra_insertion(s,pair.vbra,pair.u,x[:,:,1]),-1)
    rhs = b[:,:,4]+z*(ket_insertion(s,pair.v,pair.w,x[:,:,3])+
          bra_insertion(s,pair.vbra,pair.u,x[:,:,2])+pair.w*x[:,:,1]*pair.u'-pair.overlap*x[:,:,1])
    x[:,:,4] = solveblock(rhs,0)
    return x
end
