# Run in an otherwise quiet process, e.g. --threads=3 --gcthreads=1.
# Full evaluations use parallel=:both and BLAS=1. State construction is untimed.
# Default cases: random32,random48,random64,saved_phi4_28.
# Optional saved_shg_32 stress case can be substantially more expensive.
# Saved states are read from benchmark/legacy_states/. The shg file is not shipped:
# git show v0.1.0:data/shg_g4.0_D32_manually_added.jld > benchmark/legacy_states/shg_g4.0_D32_manually_added.jld
# RCMPS_SAMPLES=3, RCMPS_SKIP_DENSE=false, RCMPS_SKIP_TIGHT_ODE=false.
# Optional controls: RCMPS_KRYLOV_TOL, RCMPS_KRYLOV_MAXDIM,
# RCMPS_KINETIC_MAXNODES, RCMPS_LARGE_OUTPUT (new file recommended).
module LargeScaling
using RcmpsCore, LinearAlgebra, Random, Statistics, Printf, Dates, JLD
const RC=RcmpsCore
const terms=[(1.,RC.kinetic_spec(false)),(1.,RC.kinetic_spec(true))]
const fields=(:case,:D,:source,:purpose,:stage,:backend,:parallel,:julia_threads,:blas_threads,
    :window,:ode_atol,:ode_rtol,:ode_nodes,:kinetic_minnodes,:kinetic_maxnodes,
    :kinetic_atol,:kinetic_rtol,:krylov_tolerance,:krylov_maxdim,:samples,
    :min_seconds,:median_seconds,:min_bytes,:median_bytes,:min_gc_seconds,:median_gc_seconds,
    :energy,:gradient_norm,:energy_difference,:gradient_difference,:reference,
    :context_bytes_including_state,:basis_bytes,:basis_dimensions,:true_relative_residual,
    :checked_solves,:lindblad_actions,:accepted_nodes,:quadrature_energy_change,
    :quadrature_gradient_change,:status,:message)
flag(name,default=false)=lowercase(get(ENV,name,string(default))) in ("1","true","yes")
cell(x)=x isa AbstractString ? "\""*replace(x,"\""=>"\"\"")*"\"" : string(x)
function logline(logfile,message)
    println(message);flush(stdout)
    open(logfile,"a") do io
        println(io,Dates.now()," ",message)
    end
end
function row(path,base;kw...)
    values=merge(base,(;kw...))
    open(path,"a") do io
        position(io)==0 && println(io,join(string.(fields),','))
        println(io,join((cell(get(values,k,"")) for k in fields),','));flush(io)
    end
end

function state_for(label)
    source,purpose = "", ""
    s = if startswith(label,"random")
        d=parse(Int,replace(label,"random"=>""))
        rng=MersenneTwister(6100+d)
        r=randn(rng,ComplexF64,d,d)/sqrt(d)
        k=randn(rng,ComplexF64,d,d)/sqrt(d)
        source="MersenneTwister($(6100+d)); R then K; scaling 1/sqrt(D)"
        purpose="random_initial_g2_comparison"
        State(r,k+k')
    else
        file = label=="saved_phi4_28" ? "phi4_g2.0_D28_BaBajp39uG.jld" :
               label=="saved_shg_32" ? "shg_g4.0_D32_manually_added.jld" :
               throw(ArgumentError("unknown large-scaling case: $label"))
        source=joinpath(@__DIR__,"legacy_states",file)
        purpose=label=="saved_shg_32" ? "saved_shg_stress_state_evaluated_at_phi4_g2_not_g2_optimum" :
                                        "saved_phi4_g2_state"
        data=JLD.load(source)
        State(data["R"],data["K"])
    end
    fine=startswith(label,"saved")
    s.model_param.name="phi4";s.model_param.coupling=2.
    s.manifold_param.metric_reg=1e-3
    s.manifold_param.precond_reg=1e-4;s.manifold_param.precond_reg_prefactor=1e-2
    s.integrator_param.max_point=20.
    s.integrator_param.atol=fine ? 1e-13 : 1e-11
    s.integrator_param.rtol=fine ? 1e-12 : 1e-10
    s.integrator_param.n_quad=fine ? 192 : 128
    return s,source,purpose
end
function measured(f,count)
    f() # untimed warmup; each subsequent call must construct fresh solver context
    samples=[@timed f() for _ in 1:count]
    best=samples[argmin(getproperty.(samples,:time))]
    stats=(samples=count,min_seconds=best.time,median_seconds=median(getproperty.(samples,:time)),
           min_bytes=minimum(getproperty.(samples,:bytes)),median_bytes=median(getproperty.(samples,:bytes)),
           min_gc_seconds=minimum(getproperty.(samples,:gctime)),median_gc_seconds=median(getproperty.(samples,:gctime)))
    return best.value,stats
end
function finished(s,raw)
    e,c,a=raw
    e,RC.finish_gradient(s,copy(c),a)
end

# Instrument the production adaptive controller without copying its stopping rule.
# This private dispatch exists only in the benchmark process, not in package files.
struct TrackedContext{T}
    inner::T
    components::Dict{Tuple{Int,Bool},Any}
end
struct TrackedOptions{T}
    context::T
end
RC.kinetic_context(s,::Val{:scaling_diagnostic},p::TrackedOptions)=p.context
function RC.kinetic_components(s,spec,ctx::TrackedContext,grid;gradient=true)
    result=RC.kinetic_components(s,spec,ctx.inner,grid;gradient)
    ctx.components[(length(grid[1]),spec.commutator)]=result
    result
end
function diagnostic(s,angular,krylov)
    inner=RC.kinetic_context(s,Val(:krylov),krylov)
    ctx=TrackedContext(inner,Dict{Tuple{Int,Bool},Any}())
    raw=RC.kinetic_resolvent(s,terms;options=angular,backend=:scaling_diagnostic,
                            krylov_options=TrackedOptions(ctx))
    n=maximum(first(k) for k in keys(ctx.components))
    function at(nodes)
        first_energy,first_covector,first_environment=ctx.components[(nodes,false)]
        second_energy,second_covector,second_environment=ctx.components[(nodes,true)]
        finished(s,(first_energy+second_energy,first_covector+second_covector,
                    first_environment+second_environment))
    end
    accepted_energy,accepted_gradient=finished(s,raw)
    previous_energy,previous_gradient=at(n÷2)
    dimensions=join(("$(key[1] ? "commutator" : "R")_$(key[2] ? "adjoint" : "forward"):$(inner.bases[key].k)"
                     for key in sort!(collect(keys(inner.bases)))),";")
    info=(context_bytes_including_state=Base.summarysize(inner),
          basis_bytes=sum(Base.summarysize(b) for b in values(inner.bases)),
          basis_dimensions=dimensions,true_relative_residual=inner.maximum_relative_residual,
          checked_solves=inner.checked_solves,lindblad_actions=inner.actions,accepted_nodes=n,
          quadrature_energy_change=abs(accepted_energy-previous_energy),
          quadrature_gradient_change=norm(accepted_gradient-previous_gradient))
    (accepted_energy,accepted_gradient),info
end

function run_case(label,path,logfile,count,angular,krylov)
    logline(logfile,"START $label")
    s,source,purpose=state_for(label)
    p=s.integrator_param
    base=(case=label,D=s.D,source=source,purpose=purpose,julia_threads=Threads.nthreads(),
          blas_threads=BLAS.get_num_threads(),window=p.max_point,ode_atol=p.atol,ode_rtol=p.rtol,
          ode_nodes=p.n_quad,kinetic_minnodes=angular.minnodes,kinetic_maxnodes=angular.maxnodes,
          kinetic_atol=angular.atol,kinetic_rtol=angular.rtol,krylov_tolerance=krylov.tolerance,
          krylov_maxdim=krylov.maxdim)
    results=Dict{Symbol,Any}()
    for backend in (:ode,:krylov)
        stagebase=merge(base,(stage="full_phi4_g2",backend=string(backend),parallel="both"))
        try
            result,stats=measured(()->fg_phi4(s,2.;parallel=:both,kinetic_backend=backend,
                                             kinetic_options=angular,krylov_options=krylov),count)
            results[backend]=result
            reference=get(results,:ode,nothing)
            e,g=result
            row(path,merge(stagebase,stats);energy=e,gradient_norm=norm(g),
                energy_difference=isnothing(reference) ? NaN : abs(e-reference[1]),
                gradient_difference=isnothing(reference) ? NaN : norm(g-reference[2]),
                reference="full_ODE_L20_same_state_and_controls",status="ok")
            logline(logfile,"$label full $backend min=$(stats.min_seconds)s median=$(stats.median_seconds)s")
        catch error
            error isa InterruptException && rethrow()
            message=sprint(showerror,error)
            row(path,stagebase;status="failed",message)
            logline(logfile,"$label full $backend FAILED: $message")
        end
    end

    try
        result,stats=measured(()->finished(s,RC.kinetic_resolvent(s,terms;
                            options=angular,backend=:krylov,krylov_options=krylov)),count)
        row(path,merge(base,stats);stage="kinetic_pair",backend="krylov",parallel="serial",
            energy=result[1],gradient_norm=norm(result[2]),status="ok")
        logline(logfile,"$label Krylov kinetic min=$(stats.min_seconds)s median=$(stats.median_seconds)s")
    catch error
        error isa InterruptException && rethrow()
        row(path,base;stage="kinetic_pair",backend="krylov",status="failed",message=sprint(showerror,error))
        logline(logfile,"$label Krylov kinetic timing FAILED: $(sprint(showerror,error))")
    end

    kinetic_result=nothing
    try
        kinetic_result,info=diagnostic(s,angular,krylov)
        e,g=kinetic_result
        row(path,merge(base,info);stage="kinetic_adaptive_diagnostic",backend="krylov",parallel="serial",
            energy=e,gradient_norm=norm(g),status="ok",
            message="Context memory includes the referenced State; basis_bytes excludes it. Node differences are empirical, not certified bounds.")
        logline(logfile,"$label kinetic nodes=$(info.accepted_nodes), bases=$(info.basis_dimensions), residual=$(info.true_relative_residual)")
    catch error
        error isa InterruptException && rethrow()
        row(path,base;stage="kinetic_adaptive_diagnostic",backend="krylov",status="failed",message=sprint(showerror,error))
        logline(logfile,"$label kinetic diagnostics FAILED: $(sprint(showerror,error))")
    end

    used_dense=false
    if s.D<=32 && !flag("RCMPS_SKIP_DENSE")
        try
            result,stats=measured(()->finished(s,RC.kinetic_resolvent(s,terms;
                                options=angular,backend=:resolvent)),count)
            ed,gd=result
            row(path,merge(base,stats);stage="kinetic_dense_reference",backend="resolvent",parallel="serial",
                energy=ed,gradient_norm=norm(gd),reference="adaptive_krylov_kinetic_infinite_line",
                energy_difference=isnothing(kinetic_result) ? NaN : abs(ed-kinetic_result[1]),
                gradient_difference=isnothing(kinetic_result) ? NaN : norm(gd-kinetic_result[2]),status="ok")
            logline(logfile,"$label dense kinetic min=$(stats.min_seconds)s median=$(stats.median_seconds)s")
            used_dense=true
        catch error
            error isa InterruptException && rethrow()
            row(path,base;stage="kinetic_dense_reference",backend="resolvent",status="failed",message=sprint(showerror,error))
            logline(logfile,"$label dense kinetic FAILED: $(sprint(showerror,error))")
        end
    end
    # At D>32 dense physical transfer matrices are never constructed here.
    # Use one independently tightened ODE evaluation when dense is skipped/fails.
    if !used_dense && !flag("RCMPS_SKIP_TIGHT_ODE")
        tight=deepcopy(s)
        tight.integrator_param.max_point=30.
        tight.integrator_param.atol=1e-13;tight.integrator_param.rtol=1e-12
        tight.integrator_param.n_quad=256
        try
            sample=@timed RC.linear_combination(tight,terms;parallel=:both,kinetic_backend=:ode)
            er,gr=sample.value
            row(path,base;stage="kinetic_tight_ODE_reference",backend="ode",parallel="both",window=30.,
                ode_atol=1e-13,ode_rtol=1e-12,ode_nodes=256,samples=1,min_seconds=sample.time,
                min_bytes=sample.bytes,min_gc_seconds=sample.gctime,energy=er,gradient_norm=norm(gr),
                energy_difference=isnothing(kinetic_result) ? NaN : abs(er-kinetic_result[1]),
                gradient_difference=isnothing(kinetic_result) ? NaN : norm(gr-kinetic_result[2]),
                reference="adaptive_krylov_kinetic_infinite_line",status="ok",
                message="One validation evaluation; not a warmed timing comparison. Finite-window and infinite-line objectives differ.")
            logline(logfile,"$label tight L30 ODE kinetic reference completed")
        catch error
            error isa InterruptException && rethrow()
            row(path,base;stage="kinetic_tight_ODE_reference",backend="ode",status="failed",message=sprint(showerror,error))
            logline(logfile,"$label tight L30 ODE reference FAILED: $(sprint(showerror,error))")
        end
    end
    logline(logfile,"DONE $label")
end
function main()
    BLAS.set_num_threads(1)
    count=parse(Int,get(ENV,"RCMPS_SAMPLES","3"));count>=1 || throw(ArgumentError("RCMPS_SAMPLES must be positive"))
    cases=split(get(ENV,"RCMPS_LARGE_CASES","random32,random48,random64,saved_phi4_28"),',')
    angular=KineticOptions(maxnodes=parse(Int,get(ENV,"RCMPS_KINETIC_MAXNODES","512")))
    krylov=KineticKrylovOptions(tolerance=parse(Float64,get(ENV,"RCMPS_KRYLOV_TOL","1e-12")),
                               maxdim=parse(Int,get(ENV,"RCMPS_KRYLOV_MAXDIM","512")))
    stamp=Dates.format(Dates.now(),"yyyymmdd_HHMMSS")
    path=abspath(get(ENV,"RCMPS_LARGE_OUTPUT",joinpath(@__DIR__,"large_scaling","t$(Threads.nthreads())_$stamp.csv")))
    mkpath(dirname(path));logfile=path*".log"
    logline(logfile,"Large-D comparison: Julia $(VERSION), threads=$(Threads.nthreads()), BLAS=1, samples=$count, output=$path")
    logline(logfile,"Full timings hold L20 fixed; Krylov kinetics integrate the infinite line. No optimizer runs or convergence claims. Dense kinetic references are restricted to D<=32.")
    for label in cases
        try
            run_case(strip(label),path,logfile,count,angular,krylov)
        catch error
            error isa InterruptException && rethrow()
            row(path,(case=label,);stage="case_setup",status="failed",message=sprint(showerror,error))
            logline(logfile,"$label CASE FAILED: $(sprint(showerror,error))")
        end
    end
    logline(logfile,"Finished all requested cases")
end
end
abspath(PROGRAM_FILE)==abspath(@__FILE__) && LargeScaling.main()
