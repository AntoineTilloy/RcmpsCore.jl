"""Numerical controls. `max_point` is the physical half-window in units m=1;
`n_quad` is the number of Gauss nodes on each half of x=t|t|.
"""
Base.@kwdef mutable struct IntegratorParameters
    n_quad::Int = 128
    max_point::Float64 = 20.0
    atol::Float64 = 1e-11
    rtol::Float64 = 1e-10
    maxiters::Int = 1_000_000
end
Base.@kwdef mutable struct ModelParameters
    name::String = "shg"
    coupling::ComplexF64 = 1.0
end
Base.@kwdef mutable struct ManifoldParameters
    metric_reg::Float64 = 1e-3
    precond_reg_prefactor::Float64 = 1e-2
    precond_reg::Float64 = 1e-4
end

"""A left-canonical RCMPS with Q=-iK-R'R/2. Treat matrices as read-only;
use `State(R,K)` or `retraction` to change parameters and refresh the steady state.
Scratch space belongs to individual evaluations, so these may run concurrently.
The CPU implementation uses ComplexF64 throughout.
"""
mutable struct State
    D::Int
    R::Matrix{ComplexF64}
    K::Matrix{ComplexF64}
    Q::Matrix{ComplexF64}
    ρ0::Matrix{ComplexF64}
    model_param::ModelParameters
    integrator_param::IntegratorParameters
    manifold_param::ManifoldParameters
    e0::Float64
    grad::Matrix{ComplexF64}
    info::String
end

function State(R::AbstractMatrix, K::AbstractMatrix; ρ0=nothing, steady_method=:auto,
               steady_tol=1e-11)
    size(R,1) == size(R,2) > 0 || throw(ArgumentError("R must be nonempty and square"))
    size(K) == size(R) || throw(DimensionMismatch("K and R must have equal dimensions"))
    all(isfinite, R) && all(isfinite, K) || throw(ArgumentError("nonfinite state parameters"))
    norm(K-K') <= 1e-12*max(1,norm(K)) || throw(ArgumentError("K must be Hermitian"))
    r, k = Matrix{ComplexF64}(R), Matrix{ComplexF64}((K+K')/2)
    q = -im*k-r'*r/2
    rho = isnothing(ρ0) ? steady_state(q,r; method=steady_method,tol=steady_tol).rho :
          checked_density(q,r,ρ0; tol=steady_tol)
    d = size(r,1)
    return State(d,r,k,q,rho,ModelParameters(),IntegratorParameters(),ManifoldParameters(),
                 NaN,zeros(ComplexF64,d,d),"")
end

function RandomState(D::Integer; scaling=1.0, seed=1234, rng=MersenneTwister(seed), T=ComplexF64)
    D > 0 || throw(ArgumentError("D must be positive"))
    T == ComplexF64 || throw(ArgumentError("this implementation uses ComplexF64"))
    k = randn(rng,ComplexF64,D,D)
    return State(scaling*randn(rng,ComplexF64,D,D),scaling*(k+k'))
end
function RandomStateZ2(D::Integer; scaling=1.0, seed=1234, rng=MersenneTwister(seed), T=ComplexF64)
    D > 0 && iseven(D) || throw(ArgumentError("Z2 states require a positive even D"))
    T == ComplexF64 || throw(ArgumentError("this implementation uses ComplexF64"))
    d = D÷2
    k1, k2 = randn(rng,ComplexF64,d,d), randn(rng,ComplexF64,d,d)
    z = zeros(ComplexF64,d,d)
    r = [z randn(rng,ComplexF64,d,d); randn(rng,ComplexF64,d,d) z]
    return State(scaling*r,scaling*[k1+k1' z; z k2+k2'])
end

function copy_param!(dest::State, src::State)
    dest.model_param = deepcopy(src.model_param)
    dest.integrator_param = deepcopy(src.integrator_param)
    dest.manifold_param = deepcopy(src.manifold_param)
    dest.info = src.info
    return dest
end
