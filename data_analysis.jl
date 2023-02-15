using Distributed 
using Plots
@everywhere include("./src/RcmpsCore.jl")
@everywhere using .RcmpsCore 

states = RcmpsCore.get_states();
gap(x) = RcmpsCore.mass_gap(x)

function plot_mass_gap(states,g)
    D = []
    masses = []
    for state in states
        if (state.model_param.name == "phi4") & (abs(state.model_param.coupling - g)≤1e-5)
            push!(D,state.D)
            push!(masses,gap(state))
        end
    end
    scatter!(D,masses,ms=4,label=string("coupling = ",g),
        xlabel="bond dimension", ylabel="transfer matrix gap")
end


function plot_mass_gaps(states)
    g_list = [1.0,2.0]
    plot()
    for g in g_list
        plot_mass_gap(states,g)
    end
    plot!([0,32],[0.345,0.345],label="theory for g=2")
    plot!([0,32],[0.7494,0.7494],label="theory for g=1")
end

