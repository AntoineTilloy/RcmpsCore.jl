# Import a portable state directory (format 1 of gpu/src/rcmps_gpu/io.py, or an unversioned
# research directory such as an optimized_state/ of the GPU optimizer) as an rcmps_store
# checkpoint (JLD2 schema 1). rho, if given, is checked as the stationary density of R, K.
# A state whose validation_passed is false is refused; when the directory has a gradient and
# its gradient_norm, the metric norm is recomputed and must agree (rtol 1e-9), and when it
# reports gradient_target_reached, the norm must not exceed gradient_target (the checks of
# the research import of validated GPU results). No energy or optimization is evaluated,
# and an existing file is never replaced.
#
# A checkpoint holds an energy e0 only together with its gradient grad and the integration
# settings integrator_param at which both were computed: readers take a finite e0 to mean
# all three (rcmps_load, rcmps_gpu.io.read_jld2, export_state.jl), and the State's default
# grad is zero. So e0 and grad are stored only when the directory has an energy, a gradient
# and its [integration]. Otherwise e0 stays NaN, and the energy, gradient norm and settings
# that the directory does have are written into info (a warning says so).
#
# Usage: julia --project=PATH/TO/RcmpsCore.jl import_state.jl PORTABLE_DIRECTORY NEW_CHECKPOINT.jld2
@isdefined(PORTABLE_FORMAT) || include(joinpath(@__DIR__, "portable.jl"))   # once, when scripts include each other

function import_state(directory::AbstractString, output::AbstractString)
    ispath(output) && error("Refusing to replace an existing file: $output")
    p = read_portable(directory)
    p.validation_passed === false && error("$directory: the state failed its validation (validation_passed = false)")
    s = isnothing(p.rho) ? State(p.R, p.K) : State(p.R, p.K; ρ0=p.rho)
    s.model_param = portable_model(p.model, p.coupling, p.mass)
    isnothing(p.integration) || (s.integrator_param =
        IntegratorParameters(; (k => v for (k, v) in pairs(p.integration) if !isnothing(v))...))
    isnothing(p.manifold) || (s.manifold_param =
        ManifoldParameters(; (k => v for (k, v) in pairs(p.manifold) if !isnothing(v))...))
    if !isnothing(p.gradient) && !isnothing(p.gradient_norm)
        gnorm = sqrt(metric(s, p.gradient, p.gradient))
        isapprox(gnorm, p.gradient_norm; atol=1e-12, rtol=1e-9) ||
            error("$directory: the gradient's metric norm $gnorm differs from its recorded gradient_norm $(p.gradient_norm)")
        if get(p.metadata, "gradient_target_reached", false) === true
            gnorm <= p.metadata["gradient_target"] || error("$directory: the gradient exceeds its reported target")
        end
    end
    info = something(p.info, "Imported portable state $(basename(normpath(directory)))" *
        (isnothing(p.producer) ? "" : " written by $(p.producer)") * (isnothing(p.source) ? "" : "; source: $(p.source)"))
    if !any(isnothing, (p.energy, p.gradient, p.integration))
        s.e0, s.grad = p.energy, p.gradient
    elseif !all(isnothing, (p.energy, p.gradient, p.gradient_norm))
        note = unstored_evaluation(p)
        @warn "$directory: $note"
        info = isempty(info) ? note : "$info; $note"
    end
    s.info = info
    rcmps_store(s, output; verbose=false)
    return s
end

"""What a directory says about the state's evaluation when the checkpoint cannot hold it as e0."""
function unstored_evaluation(p)
    i = p.integration
    known = String[]
    isnothing(p.energy) || push!(known, "energy = $(p.energy)")
    isnothing(p.gradient_norm) || push!(known, "gradient_norm = $(p.gradient_norm)")
    isnothing(p.gradient) || push!(known, "a stored gradient")
    push!(known, isnothing(i) ? "settings unknown" :
          "computed at n_quad = $(i.n_quad), max_point = $(i.max_point), atol = $(i.atol), rtol = $(i.rtol)")
    absent = [name for (name, value) in (("energy", p.energy), ("gradient", p.gradient), ("integration settings", i))
               if isnothing(value)]
    return "not stored as e0 (the directory has no $(join(absent, ", ", " or ")); a checkpoint holds an energy " *
           "only with its gradient and integration settings): " * join(known, ", ")
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    length(ARGS) == 2 ||
        error("Usage: julia --project=PATH/TO/RcmpsCore.jl import_state.jl PORTABLE_DIRECTORY NEW_CHECKPOINT.jld2")
    s = import_state(ARGS...)
    println("Imported D=$(s.D) $(s.model_param.name) g=$(real(s.model_param.coupling)) e0=$(s.e0) to $(ARGS[2])")
end
