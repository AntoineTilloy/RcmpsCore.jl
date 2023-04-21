using Distributed 
using Revise

push!(LOAD_PATH, "./Projects/")
#@everywhere include("./src/RcmpsCore.jl")
@everywhere using RcmpsCore 

x = RandomStateZ2(12);
x.model_param.name = "shg"
x.model_param.coupling = 2.0

x.manifold_param.metric_reg =1e-3
x.manifold_param.precond_reg_prefactor = 1e-5


x_new = optimize_state(x,method="LBFGS",n_iter = 100)