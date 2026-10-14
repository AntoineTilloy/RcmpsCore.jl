# Export an rcmps_store checkpoint (JLD2 schema 1) as a portable state directory, format 1
# (gpu/src/rcmps_gpu/io.py): R, K, the stationary density rho of the loaded State and, if the
# state was evaluated (finite e0), its energy and gradient; model, integration and manifold
# parameters and info from the checkpoint. Nothing is evaluated besides rho and the metric
# norm of the gradient. The source recorded is the checkpoint's file name, not its path.
#
# Usage: julia --project=PATH/TO/RcmpsCore.jl export_state.jl CHECKPOINT.jld2 NEW_DIRECTORY
@isdefined(PORTABLE_FORMAT) || include(joinpath(@__DIR__, "portable.jl"))   # once, when scripts include each other

function export_state(checkpoint::AbstractString, directory::AbstractString)
    ispath(directory) && error("Refusing to overwrite $directory: choose a new directory")
    s = rcmps_load(checkpoint)   # recomputes and checks rho
    evaluated = isfinite(s.e0)
    evaluated && !all(isfinite, s.grad) && error("$checkpoint: finite e0 with a nonfinite gradient")
    fields(x) = (; (key => getfield(x, key) for key in fieldnames(typeof(x)))...)
    write_portable(directory; R=s.R, K=s.K, rho=s.ρ0, gradient=evaluated ? s.grad : nothing,
                   model=s.model_param.name, coupling=s.model_param.coupling, mass=portable_mass(s.model_param),
                   energy=evaluated ? s.e0 : nothing,
                   gradient_norm=evaluated ? sqrt(metric(s, s.grad, s.grad)) : nothing,
                   producer="RcmpsCore $(pkgversion(RcmpsCore)), gpu/julia/export_state.jl",
                   source=basename(checkpoint), info=isempty(s.info) ? nothing : s.info,
                   integration=fields(s.integrator_param), manifold=fields(s.manifold_param))
    return s
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    length(ARGS) == 2 ||
        error("Usage: julia --project=PATH/TO/RcmpsCore.jl export_state.jl CHECKPOINT.jld2 NEW_DIRECTORY")
    s = export_state(ARGS...)
    println("Exported D=$(s.D) $(s.model_param.name) g=$(real(s.model_param.coupling)) e0=$(s.e0) to $(ARGS[2])")
end
