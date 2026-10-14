# Round trip of the portable format on the shared D = 16 fixture, in a temporary directory:
# export_state.jl of test/fixtures/phi4_g2_D16_z2_state.jld2 reproduces the committed export
# test/fixtures/phi4_g2_D16_z2_portable (R, K, gradient and metadata exactly, rho to 1e-12,
# since its last bits depend on the BLAS), and import_state.jl of either export gives back
# the checkpoint: R, K, rho (recomputed by rcmps_load), model, coupling, integration and
# manifold parameters, e0, gradient and info, all exactly. Prints one line per check.
#
# Usage: julia --project=PATH/TO/RcmpsCore.jl check_portable.jl [FIXTURES_DIRECTORY]
include(joinpath(@__DIR__, "export_state.jl"))
include(joinpath(@__DIR__, "import_state.jl"))

function check_portable(fixtures::AbstractString)
    checkpoint = joinpath(fixtures, "phi4_g2_D16_z2_state.jld2")
    committed = joinpath(fixtures, "phi4_g2_D16_z2_portable")
    original = rcmps_load(checkpoint)
    fields(x) = [getfield(x, key) for key in fieldnames(typeof(x))]
    failures = String[]
    check(name, ok) = (println(ok ? "ok    " : "FAILED", " ", name); ok || push!(failures, name))
    mktempdir() do tmp
        exported = joinpath(tmp, "export")
        export_state(checkpoint, exported)
        for name in ("R.bin", "K.bin", "gradient.bin", "metadata.toml")
            check("export $name equals the committed one", read(joinpath(exported, name)) == read(joinpath(committed, name)))
        end
        rho = read_portable(committed).rho
        check("export rho.bin equals the committed one to 1e-12", norm(read_portable(exported).rho - rho, Inf) <= 1e-12)
        check("committed rho is the state's to 1e-12", norm(rho - original.ρ0, Inf) <= 1e-12)
        for (label, directory) in (("committed", committed), ("exported", exported))
            path = joinpath(tmp, "$label.jld2")
            import_state(directory, path)
            s = rcmps_load(path)
            check("$label: R, K, rho exact", s.R == original.R && s.K == original.K && s.ρ0 == original.ρ0)
            check("$label: model, coupling exact", s.model_param.name == original.model_param.name &&
                  s.model_param.coupling === original.model_param.coupling &&
                  portable_mass(s.model_param) == portable_mass(original.model_param))
            check("$label: integration, manifold exact", fields(s.integrator_param) == fields(original.integrator_param) &&
                  fields(s.manifold_param) == fields(original.manifold_param))
            check("$label: e0, gradient, info exact", s.e0 === original.e0 && s.grad == original.grad &&
                  s.info == original.info)
        end
    end
    isempty(failures) || error("$(length(failures)) checks failed")
    println("All portable round-trip checks passed")
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    length(ARGS) <= 1 || error("Usage: julia --project=PATH/TO/RcmpsCore.jl check_portable.jl [FIXTURES_DIRECTORY]")
    check_portable(isempty(ARGS) ? joinpath(@__DIR__, "..", "..", "test", "fixtures") : ARGS[1])
end
