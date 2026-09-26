"""Write an explicit, versioned JLD2 schema via an atomic rename.
The path overload takes a full path, not a name implicitly relative to ./data.
"""
function rcmps_store(s::State,path::AbstractString;verbose=true)
    path = abspath(path)
    mkpath(dirname(path))
    temp,io = mktemp(dirname(path))
    close(io)
    try
        JLD2.jldopen(temp,"w") do f
            f["schema_version"] = 1
            f["R"],f["K"] = s.R,s.K
            f["model"],f["coupling"] = s.model_param.name,s.model_param.coupling
            f["e0"],f["grad"],f["info"] = s.e0,s.grad,s.info
            f["integration"] = (;(key=>getfield(s.integrator_param,key) for key in fieldnames(IntegratorParameters))...)
            f["manifold"] = (;(key=>getfield(s.manifold_param,key) for key in fieldnames(ManifoldParameters))...)
        end
        mv(temp,path;force=true)
    finally
        isfile(temp) && rm(temp)
    end
    verbose && @info "Saved RCMPS" path
    return path
end
function rcmps_store(s::State;directory="data",id=randstring(10),verbose=true)
    name = "$(s.model_param.name)_g$(s.model_param.coupling)_D$(s.D)_$(id).jld2"
    return rcmps_store(s,joinpath(directory,name);verbose)
end
function rcmps_load(path::AbstractString)
    data = JLD2.load(path)
    get(data,"schema_version",0) == 1 || throw(ArgumentError("unsupported schema; migrate old JLD with examples/migrate_legacy.jl"))
    s = State(data["R"],data["K"]) # Recompute and check rho, never trust a serialized cache.
    s.model_param = ModelParameters(name=data["model"],coupling=data["coupling"])
    s.integrator_param = IntegratorParameters(;data["integration"]...)
    s.manifold_param = ManifoldParameters(;data["manifold"]...)
    s.e0,s.grad,s.info = data["e0"],data["grad"],data["info"]
    return s
end
get_states(directory="data") = [rcmps_load(joinpath(directory,f)) for f in sort(readdir(directory)) if endswith(f,".jld2")]
function select_best(states)
    best = Dict{Tuple{String,ComplexF64,Int},State}()
    for s in states
        isfinite(s.e0) || continue
        key = (s.model_param.name,s.model_param.coupling,s.D)
        if !haskey(best,key) || s.e0 < best[key].e0
            best[key] = s
        end
    end
    return collect(values(best))
end
function get_exact_state(states,model,coupling,d;atol=1e-7)
    candidates = filter(s->s.model_param.name==model && s.D==d && isfinite(s.e0) &&
                          abs(s.model_param.coupling-coupling)<=atol,states)
    return isempty(candidates) ? nothing : argmin(s->s.e0,candidates)
end
