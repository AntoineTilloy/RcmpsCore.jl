using Parameters
using Random 
using LinearAlgebra

@with_kw mutable struct IntegratorParameters
    n_quad::Int64 = 1000
    max_point::Real = 20.0
    min_point::Real = 1e-15
    atol::Real = 1e-13
    rtol::Real = 1e-11
end

@with_kw mutable struct ModelParameters
    name::String = "shg"
    coupling::Number = 1.0
end

@with_kw mutable struct ManifoldParameters
    metric_reg::Real = 1e-2
    precond_reg_prefactor::Real = 1e-2
    precond_reg::Real = 1e-4
end

mutable struct State{T<:AbstractMatrix}
    D::Int
   
    R::T 
    Rd::T
    K::T
    Q::T
    Qd::T
    QRcomm::T
    RdQdcomm::T
    ρ0::T

    buffer1::T
    buffer2::T
    buffer3::T
    
    model_param::ModelParameters
    integrator_param::IntegratorParameters
    manifold_param::ManifoldParameters

    e0::Real
    grad::T
    info::String

    State(T)=new{T}() 
end



#Constructors for Rcmps state structure

#Initialize a state defined by matrices R and K, K self adjoint
function State(R::AbstractMatrix,K::AbstractMatrix ; ρ0=nothing)
    @assert norm(K-K') ≤ 10*sqrt(eps(real(eltype(R)))) "K is not self-adjoint"
    @assert typeof(R) == typeof(K) "K and R need to be of the same type"

    state=State(typeof(R))
    D=size(R)[1]
    state.D=D
    state.R=deepcopy(R)
    state.K=deepcopy(K)
    state.Rd=deepcopy(R')
    state.Q= -1im*K-0.5*adjoint(R)*R
    state.Qd=deepcopy(state.Q')
    state.QRcomm=state.Q*state.R - state.R*state.Q
    state.RdQdcomm=deepcopy(state.QRcomm')

    state.buffer1=deepcopy(K)
    state.buffer2=deepcopy(K)
    state.buffer3=deepcopy(K)
    
    state.grad=one(R)

    if typeof(ρ0) == Nothing
        state.ρ0=get_ρ0(state)
    else
        state.ρ0=ρ0
    end

    state.info="_"
    state.model_param=ModelParameters() 
    state.integrator_param=IntegratorParameters() 
    state.manifold_param=ManifoldParameters()

    state.e0=1e10
    return state
end

function RandomState(D::Int;scaling=1.0,seed=1234,T=ComplexF64)
    if seed !=1234
        Random.seed!(seed)
    end
    K=randn(T,D,D)
    K=K+K'
    R=randn(T,D,D)
    return State(scaling*R,scaling*K)
end

function RandomStateZ2(D::Int;scaling=1.0,seed=1234,T=ComplexF64)
    d = floor(Int,D/2)
    if seed !=1234
        Random.seed!(seed)
    end
    K1=randn(T,d,d)
    K1=K1+K1'
    R1=randn(T,d,d)

    K2=randn(T,d,d)
    K2=K2+K2'
    R2=randn(T,d,d)

    zer=zeros(T,d,d)

    R=[zer R1 ; R2 zer]
    K=[K1 zer ; zer K2]

    return State(scaling*R,scaling*K)
end

function copy_param!(state1,state2)
    state1.model_param=state2.model_param
    state1.integrator_param=state2.integrator_param
    state1.manifold_param=state2.manifold_param
    state1.e0=state2.e0
    state1.info=state2.info
end
