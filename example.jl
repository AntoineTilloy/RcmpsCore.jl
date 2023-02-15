using Distributed 
@everywhere include("./src/RcmpsCore.jl")
@everywhere using .RcmpsCore 

x = RandomStateZ2(6,scaling=1.7);
x.model_param.name = "sg"
x.model_param.coupling = 4.6

x.manifold_param.metric_reg = 1e-3
x.manifold_param.precond_reg_prefactor = 1e-2


x_new = optimize_state(x,method="CG",n_iter = 100)

rcmps_store(x_new,id="test")

x_new = optimize_state(x,method="LBFGS",n_iter = 2000)

rcmps_store(x_new,id="test")

@info "done"