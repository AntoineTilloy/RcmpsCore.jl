# Julia excitation levels of phi4 states without Z2 symmetry (the sector all), at the settings
# `gpu_comparison` of test/fixtures/excitation_references.toml: the settings of the GPU
# command line rcmps-gpu-excitations (window 30, n_quad 512, tolerances 1e-13/1e-12, 256 kinetic
# nodes; validation at window 40, n_quad 1024, 1e-14/1e-13, 512 nodes; a search space of
# min(160, n) vectors with thick restarts keeping 12), with the dense resolvent kinetic backend.
# The GPU side compares with them through the command line (gpu/README.md, "Excitations of
# states without symmetry"). Writes one TOML file with a [[levels]] entry per level.
#
# Usage, from the repository root:
#   julia --project=. -t 2 gpu/qualification/excitations/nonz2_levels.jl NEW_FILE.toml STATE P NEV [STATE P NEV]...
# STATE: an rcmps_store checkpoint (.jld2) or a portable state directory (gpu/julia/portable.jl).
@isdefined(PORTABLE_FORMAT) || include(joinpath(@__DIR__, "..", "..", "julia", "portable.jl"))
using Printf, SHA
BLAS.set_num_threads(1)

const INTEGRATION = IntegratorParameters(n_quad=512, max_point=30.0, atol=1e-13, rtol=1e-12)
const REFINED = IntegratorParameters(n_quad=1024, max_point=40.0, atol=1e-14, rtol=1e-13)
const FOUR = (:standard_energy, :refined_energy, :bilinear_energy, :refined_bilinear_energy)

function load(path)
    endswith(path, ".jld2") && return rcmps_load(path)
    p = read_portable(path)
    s = isnothing(p.rho) ? State(p.R, p.K) : State(p.R, p.K; ρ0=p.rho)
    s.model_param = portable_model(p.model, p.coupling, p.mass)
    return s
end

function digest(path)
    isdir(path) || return bytes2hex(open(sha256, path))
    bytes2hex(sha256(join(bytes2hex(open(sha256, joinpath(path, f))) for f in sort(readdir(path)))))
end

function levels(path, p, nev)
    s = load(path)
    s.integrator_param = deepcopy(INTEGRATION)
    coordinates = TangentCoordinates(s; sector=:auto, parity=:auto, gauge=:density, metric_tol=0.0)
    n = length(coordinates)
    davidson = DavidsonOptions(tol=1e-7, maxops=200, maxspace=min(160, n), retain=12, refresh=false,
                               hermitian_tol=1e-7, stall_window=20, stall_minops=40, stall_rtol=0.1,
                               stall_residual_limit=1e-5, stall_energy_atol=1e-9)
    model = ExcitationModel(Symbol(s.model_param.name), real(s.model_param.coupling))
    tick = time()
    r = excitation_eigsolve(s, p; model, sector=:auto, parity=:auto, gauge=:density, metric_tol=0.0, nev,
        kinetic_nodes=256, refined=REFINED, refined_kinetic_nodes=512, preconditioner=:kinetic,
        preconditioner_nodes=8, start=:auto, davidson, validate=true, validation_tol=1e-7,
        acceptance_residual=1e-5, refine_steps=0, environment_tol=1e-10, max_bond=64,
        kinetic_backend=:resolvent, parallel=:both)
    seconds = time() - tick
    out = Dict{String,Any}[]
    for k in 1:nev
        v = r.validation[k]
        push!(out, Dict{String,Any}("state" => abspath(path), "state_digest" => digest(path), "D" => s.D,
            "coupling" => real(s.model_param.coupling), "momentum" => Float64(p), "nev" => nev, "level" => k,
            "sector" => string(coordinates.sector), "dimension" => r.dimension,
            "validated_energy" => r.validated_energies[k], "ritz_energy" => r.energies[k],
            "residual" => r.residuals[k], "four_energies" => [Float64(getfield(v, f)) for f in FOUR],
            "energy_spread" => v.energy_spread, "accepted" => v.accepted, "converged" => r.converged,
            "numops" => r.numops, "action_calls" => r.action_calls, "start" => string(r.start),
            "stop_reason" => string(r.stop_reason), "wall_seconds" => seconds,
            "julia_threads" => Threads.nthreads(), "rcmpscore_version" => string(pkgversion(RcmpsCore))))
        @printf("%s p=%g level %d: validated %.15f ritz %.15f spread %.2e accepted %s (%d ops, %.0f s)\n",
                basename(normpath(path)), p, k, r.validated_energies[k], r.energies[k], v.energy_spread,
                v.accepted, r.numops, seconds)
        flush(stdout)
    end
    return out
end

function main(arguments)
    length(arguments) >= 4 && (length(arguments) - 1) % 3 == 0 ||
        error("usage: nonz2_levels.jl NEW_FILE.toml STATE P NEV [STATE P NEV]...")
    output = arguments[1]
    ispath(output) && error("Refusing to overwrite $output")
    entries = Dict{String,Any}[]
    for i in 2:3:length(arguments)
        append!(entries, levels(arguments[i], parse(Float64, arguments[i + 1]), parse(Int, arguments[i + 2])))
        open(output * ".partial", "w") do io
            TOML.print(io, Dict("settings" => "gpu_comparison of test/fixtures/excitation_references.toml",
                                "levels" => entries); sorted=true)
        end
    end
    mv(output * ".partial", output)
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && main(ARGS)
