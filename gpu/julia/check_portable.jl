# Round trip of the portable format on the shared D = 16 fixture, in a temporary directory:
# export_state.jl of test/fixtures/phi4_g2_D16_z2_state.jld2 reproduces the committed export
# test/fixtures/phi4_g2_D16_z2_portable (R, K, gradient and metadata exactly, rho to 1e-12,
# since its last bits depend on the BLAS), and import_state.jl of either export gives back
# the checkpoint: R, K, rho (recomputed by rcmps_load), model, coupling, integration and
# manifold parameters, e0, gradient and info, all exactly. Directories that rcmps_gpu.io
# refuses (a format_version that is not an integer, an empty model name) are refused too.
# Prints one line per check.
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
        # Refused here as by rcmps_gpu.io (tests/test_io.py).
        function edited(name, old, new)
            directory = joinpath(tmp, name)
            cp(committed, directory)
            path = joinpath(directory, "metadata.toml")
            text = read(path, String)
            count(old, text) == 1 || error("$old is not once in the committed metadata.toml")
            write(path, replace(text, old => new))
            return directory
        end
        refused(f, message) = try f(); false catch e; occursin(message, sprint(showerror, e)) end
        check("format_version = 1.0 is refused", refused(() -> read_portable(edited("version",
              "format_version = 1\n", "format_version = 1.0\n")), "unsupported rcmps-portable-state format_version 1.0"))
        check("an empty model name is refused", refused(() -> read_portable(edited("model",
              "model = \"phi4\"", "model = \"\"")), "the model name is empty"))
        check("an empty model name is not written", refused(() -> write_portable(joinpath(tmp, "unwritten");
              R=original.R, K=original.K, model="", coupling=2.0), "the model name is empty") &&
              !ispath(joinpath(tmp, "unwritten")))
    end
    isempty(failures) || error("$(length(failures)) checks failed")
    println("All portable round-trip checks passed")
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    length(ARGS) <= 1 || error("Usage: julia --project=PATH/TO/RcmpsCore.jl check_portable.jl [FIXTURES_DIRECTORY]")
    check_portable(isempty(ARGS) ? joinpath(@__DIR__, "..", "..", "test", "fixtures") : ARGS[1])
end
