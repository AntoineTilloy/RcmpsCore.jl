using LinearMaps 
using KrylovKit

function robust_inv(M::Matrix,m_reg)
    return inv(M+m_reg*I)
end

# A weakness of the present algorithm is in the computation of the fixed point
# ρ0 that is fast but can fail, and if so we go to a sligthly slower method.

function is_ρ0_sound(state::State,ρ0::AbstractMatrix)
    ε2=norm(lindblad(state,ρ0))
    return ((ε2 <1e-5))
end

function get_ρ0(state::State)
    #Start with cheepest evaluation of the stationary state
    ρ0=get_ρ0_krylov(state)

    if (is_ρ0_sound(state,ρ0)==false)
        @warn "Trying to evaluate ρ0 with Krylov failed, trying to minimize (L'L)"
        ρ0=get_ρ0_krylov_squared(state)
        if (is_ρ0_sound(state,ρ0)==false)
            @error "We tried everything, ρ0 still unsound"
        end
    end
    return ρ0
end

function get_ρ0_krylov(state::State)
    D=state.D
    ρ0_init=0*state.R + (1/D) * I
    ρ0_init=reshape(ρ0_init,D*D)
    lind(ρ)=reshape(lindblad(state,reshape(ρ,(D,D))),D*D)
    lind_adj(ρ)=reshape(lindblad_adjoint(state,reshape(ρ,(D,D))),D*D)
    Lb=LinearMap(lind,lind_adj,D*D)
    maxdim=min(500,D^2)
    λ,ρ0,info=eigsolve(Lb,ρ0_init,1,:LR,maxiter=100000,krylovdim=maxdim)
    ρ0=reshape(ρ0[1],(D,D))
    ρ0=0.5(ρ0+adjoint(ρ0))
    ρ0= ρ0/tr(ρ0)
    return ρ0
end

function get_ρ0_krylov_squared(state::State)
    D=state.D
    ρ0_init=0*state.R + (1/D) * I
    ρ0_init=reshape(ρ0_init,D*D)
    lind(ρ)=reshape(lindblad(state,reshape(ρ,(D,D))),D*D)
    lind_adj(ρ)=reshape(lindblad_adjoint(state,reshape(ρ,(D,D))),D*D)
    Lb=LinearMap(lind,lind_adj,D*D)
    maxdim=min(400,D^2)
    λ,ρ0,info=eigsolve(Lb'*Lb,ρ0_init,1,:SR,maxiter=100000,krylovdim=maxdim)
    ρ0=reshape(ρ0[1],(D,D))
    ρ0=0.5(ρ0+adjoint(ρ0))
    ρ0= ρ0/tr(ρ0)
    return ρ0
end

function entanglement_entropy(x::State;alpha = 1.0)
    if alpha ≈ 1.0
        S=- real(tr(x.ρ0*log(x.ρ0)))
    else
        S = real( 1/(1-alpha) *log(tr(exp(log(x.ρ0)*alpha))))
    end
    return S
end

function generate_T(x::State)
    id=Matrix{ComplexF64}(I,x.D,x.D) 
    T=kron(x.Q,id) + kron(id,x.Qd) + kron(x.R,x.Rd)
    return T 
end

function mass_gap(x::State)
    T=generate_T(x)
    lb=eigvals(T)
    gap=sort(-real.(lb))[2]
    return gap
end



function zeros_same_type(M,dims::Vararg{Union{Core.Integer,Base.AbstractUnitRange}})
 
    return zeros(eltype(M),dims) 

end