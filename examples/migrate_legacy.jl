# Run in the benchmark environment, which supplies JLD for reading historical files.
# julia --project=benchmark examples/migrate_legacy.jl benchmark/legacy_states/phi4_g2.0_D2_.jld examples/output/migrated.jld2
# rcmps_store creates missing parent directories of the destination.
using RcmpsCore, JLD
length(ARGS) in (2,3) || error("usage: migrate_legacy.jl input.jld output.jld2 [massmod]")
isfile(ARGS[2]) && error("destination already exists: $(ARGS[2])")
data = JLD.load(ARGS[1])
s = State(data["R"],data["K"])
s.model_param = ModelParameters(name=data["model"],coupling=data["coupling"])
s.info = get(data,"info","")
# Historical energies and gradients used an incomplete derivative. Preserve the
# parameters and metadata, and recompute objective/gradient for the current model.
s.model_param.name == "phi4_m" && length(ARGS) != 3 && error("historical phi4_m files require an explicit massmod argument")
massmod = length(ARGS) == 3 ? parse(Float64,ARGS[3]) : 1.0
energy,gradient = RcmpsCore.energy_function(s,massmod)(s)
s.e0,s.grad = energy,gradient
rcmps_store(s,ARGS[2])
