using Distributed 
@everywhere include("./src/RcmpsCore.jl")
@everywhere using .RcmpsCore 

x = RandomState(12);
x.model_param.name = "shg"
x.model_param.coupling = 2.0

x.manifold_param.metric_reg = 0.0
x.manifold_param.precond_reg_prefactor = 0.0


x_new = optimize_state(x,method="LBFGS",n_iter = 100)