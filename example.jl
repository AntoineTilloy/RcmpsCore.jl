using Distributed 
@everywhere include("./src/RcmpsCore.jl")
@everywhere using .RcmpsCore 

x = RandomState(18);
x.model_param.name = "phi4"
x.model_param.coupling = 2.7

x.manifold_param.metric_reg = 1e-2
x.manifold_param.precond_reg_prefactor = 1e-2


x_new = optimize_state(x,method="LBFGS",n_iter = 700)