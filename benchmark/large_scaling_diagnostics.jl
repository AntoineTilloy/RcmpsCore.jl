# Rerun corrected diagnostic bookkeeping only. No timed evaluator/reference is
# repeated and no original CSV is modified. Use the same RCMPS_LARGE_CASES,
# RCMPS_KRYLOV_TOL, RCMPS_KRYLOV_MAXDIM, RCMPS_KINETIC_MAXNODES as the timing run.
# RCMPS_DIAGNOSTICS_OUTPUT optionally names a NEW output CSV (existing rejected).
include("large_scaling.jl")
using RcmpsCore, LinearAlgebra, Dates

function rerun_diagnostics()
    BLAS.set_num_threads(1)
    cases=split(get(ENV,"RCMPS_LARGE_CASES","random32,random48,random64,saved_phi4_28,saved_shg_32"),',')
    angular=KineticOptions(maxnodes=parse(Int,get(ENV,"RCMPS_KINETIC_MAXNODES","512")))
    krylov=KineticKrylovOptions(tolerance=parse(Float64,get(ENV,"RCMPS_KRYLOV_TOL","1e-12")),
                               maxdim=parse(Int,get(ENV,"RCMPS_KRYLOV_MAXDIM","512")))
    stamp=Dates.format(Dates.now(),"yyyymmdd_HHMMSS")
    path=abspath(get(ENV,"RCMPS_DIAGNOSTICS_OUTPUT",
        joinpath(@__DIR__,"large_scaling","corrected_diagnostics_$stamp.csv")))
    ispath(path) && throw(ArgumentError("refusing to overwrite existing diagnostics: $path"))
    mkpath(dirname(path));logfile=path*".log"
    LargeScaling.logline(logfile,"Corrected diagnostics only: Julia $(VERSION), threads=$(Threads.nthreads()), BLAS=1; original timing data preserved")
    for entry in cases
        label=strip(entry)
        try
            state,source,purpose=LargeScaling.state_for(label)
            parameters=state.integrator_param
            base=(case=label,D=state.D,source=source,purpose=purpose,
                stage="corrected_kinetic_adaptive_diagnostic",backend="krylov",parallel="serial",
                julia_threads=Threads.nthreads(),blas_threads=BLAS.get_num_threads(),
                window=parameters.max_point,ode_atol=parameters.atol,ode_rtol=parameters.rtol,
                ode_nodes=parameters.n_quad,kinetic_minnodes=angular.minnodes,
                kinetic_maxnodes=angular.maxnodes,kinetic_atol=angular.atol,kinetic_rtol=angular.rtol,
                krylov_tolerance=krylov.tolerance,krylov_maxdim=krylov.maxdim)
            (kinetic_energy,kinetic_gradient),info=LargeScaling.diagnostic(state,angular,krylov)
            isfinite(kinetic_energy) && all(isfinite,kinetic_gradient) ||
                throw(SolverFailure("nonfinite corrected kinetic diagnostic"))
            LargeScaling.row(path,merge(base,info);energy=kinetic_energy,
                gradient_norm=norm(kinetic_gradient),status="ok",
                message="Corrected diagnostic closure bookkeeping; full kinetic-pair energy. Original timed evaluations unchanged. Context bytes are retained memory including State, not peak memory.")
            LargeScaling.logline(logfile,"$label corrected kinetic E=$kinetic_energy gradient_norm=$(norm(kinetic_gradient)) nodes=$(info.accepted_nodes) energy_change=$(info.quadrature_energy_change) gradient_change=$(info.quadrature_gradient_change)")
        catch error
            error isa InterruptException && rethrow()
            LargeScaling.row(path,(case=label,);stage="corrected_kinetic_adaptive_diagnostic",
                backend="krylov",status="failed",message=sprint(showerror,error))
            LargeScaling.logline(logfile,"$label corrected diagnostic FAILED: $(sprint(showerror,error))")
        end
    end
    LargeScaling.logline(logfile,"Corrected diagnostics complete: $path")
end
abspath(PROGRAM_FILE)==abspath(@__FILE__) && rerun_diagnostics()
