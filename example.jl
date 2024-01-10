using Distributed 
#@everywhere using Revise

#push!(LOAD_PATH, "./Projects/")
@everywhere include("./src/RcmpsCore.jl")
@everywhere using .RcmpsCore 

x = RandomStateZ2(32,scaling=1);
x.model_param.name = "phi4_m"
x.model_param.coupling = 1.0
massmod=1.0

x.manifold_param.metric_reg =1e-1
x.manifold_param.precond_reg_prefactor = 1e-4

x.integrator_param.atol=1e-13
x.integrator_param.rtol=1e-12
x.integrator_param.max_point = 20
x.integrator_param.min_point = 1e-16
x.integrator_param.n_quad = 4000

#x_new = optimize_state(x,method="GD",n_iter = 30,massmod=massmod)

# x_new.manifold_param.metric_reg =1e-1
# x_new.manifold_param.precond_reg_prefactor = 1e-3


# x_new.integrator_param.min_point = 1e-16
# x_new.integrator_param.n_quad = 4000
x_new=x
x_new = optimize_state(x_new,method="LBFGS",n_iter = 600,massmod=massmod)