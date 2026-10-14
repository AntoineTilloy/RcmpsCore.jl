# Import a portable state directory (format 1 of gpu/src/rcmps_gpu/io.py, or an unversioned
# research directory such as an optimized_state/ of the GPU optimizer) as an rcmps_store
# checkpoint (JLD2 schema 1). rho, if given, is checked as the stationary density of R, K.
# A state whose validation_passed is false is refused; when the directory has a gradient and
# its gradient_norm, the metric norm is recomputed and must agree (rtol 1e-9), and when it
# reports gradient_target_reached, the norm must not exceed gradient_target (the checks of
# the research import of validated GPU results). No energy or optimization is evaluated,
# and an existing file is never replaced.
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
    isnothing(p.energy) || (s.e0 = p.energy)
    isnothing(p.gradient) || (s.grad = p.gradient)
    if !isnothing(p.gradient) && !isnothing(p.gradient_norm)
        gnorm = sqrt(metric(s, s.grad, s.grad))
        isapprox(gnorm, p.gradient_norm; atol=1e-12, rtol=1e-9) ||
            error("$directory: the gradient's metric norm $gnorm differs from its recorded gradient_norm $(p.gradient_norm)")
        if get(p.metadata, "gradient_target_reached", false) === true
            gnorm <= p.metadata["gradient_target"] || error("$directory: the gradient exceeds its reported target")
        end
    end
    s.info = something(p.info, "Imported portable state $(basename(normpath(directory)))" *
        (isnothing(p.producer) ? "" : " written by $(p.producer)") * (isnothing(p.source) ? "" : "; source: $(p.source)"))
    rcmps_store(s, output; verbose=false)
    return s
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    length(ARGS) == 2 ||
        error("Usage: julia --project=PATH/TO/RcmpsCore.jl import_state.jl PORTABLE_DIRECTORY NEW_CHECKPOINT.jld2")
    s = import_state(ARGS...)
    println("Imported D=$(s.D) $(s.model_param.name) g=$(real(s.model_param.coupling)) e0=$(s.e0) to $(ARGS[2])")
end
