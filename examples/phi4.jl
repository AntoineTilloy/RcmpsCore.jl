using RcmpsCore
using LinearAlgebra
BLAS.set_num_threads(1) # Good starting point for small bond dimensions; benchmark larger D.
s = RandomStateZ2(8;seed=1234,scaling=.2)
s.model_param.name = "phi4"
s.model_param.coupling = 2.0
result = optimize_state(s;n_iter=20,method="LBFGS",diagnostics=true)
println((energy=result.energy,gradient_norm=result.gradient_norm,converged=result.converged))
rcmps_store(result.state,joinpath(@__DIR__,"output","phi4.jld2"))
