using Distributed 
#@everywhere using Revise

#push!(LOAD_PATH, "./Projects/")
@everywhere include("./src/RcmpsCore.jl")
@everywhere using .RcmpsCore 

x = RandomStateZ2(16,scaling=1.5,seed=17);
x.model_param.name = "shg"
x.model_param.coupling = 4.0

x.manifold_param.metric_reg =1e-2
x.manifold_param.precond_reg_prefactor = 1e-3

x.integrator_param.atol=1e-13
x.integrator_param.rtol=1e-12
x.integrator_param.max_point = 20
x.integrator_param.min_point = 1e-16
x.integrator_param.n_quad = 2000


x_new = optimize_state(x,method="LBFGS",n_iter = 600)