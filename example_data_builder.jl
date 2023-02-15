using Distributed 
@everywhere include("./src/RcmpsCore.jl")
@everywhere using .RcmpsCore 

coupling = 2.0

bond_dimensions = Array(18:2:20)
@distributed for D in bond_dimensions
    x = RandomState(D);
    x.model_param.name = "phi4"
    x.model_param.coupling = coupling

    x.manifold_param.metric_reg = 1e-1
    x.manifold_param.precond_reg_prefactor = 1e-2


    x_new = optimize_state(x,method="LBFGS",n_iter = 1000)

    rcmps_store(x_new)
end

@info "done"