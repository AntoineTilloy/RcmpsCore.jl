# Short independent optimization runs; fixed precision and LBFGS history throughout.
using RcmpsCore, LinearAlgebra, Printf, Statistics
import OptimKit

const angular = KineticOptions(maxnodes=1024)
const krylov = KineticKrylovOptions(maxdim=1024)
objective(s) = fg_phi4(s, 2.; parallel=:both, kinetic_backend=:krylov,
                       kinetic_options=angular, krylov_options=krylov)

function run_case(initial, threads, tag, output)
    BLAS.set_num_threads(threads)
    BLAS.get_num_threads() == threads || error("BLAS thread setting did not apply")
    println("Warmup BLAS=", threads, " tag=", tag); flush(stdout)
    warm = @timed objective(initial)
    @printf("Warmup %.3fs E=%.14f\n", warm.time, warm.value[1]); flush(stdout)
    s = deepcopy(initial)
    laststate = Ref(s)
    calls = Ref(0)
    iteration = Ref(0)
    firstcall = Ref(true)
    fg_times = Float64[]
    accepted_times = Float64[]
    trajectory = joinpath(output, tag * "_trajectory.csv")
    isfile(trajectory) && error("Refusing to overwrite $trajectory")
    GC.gc()
    elapsed = 0.0
    open(trajectory, "w") do io
        println(io, "iteration,evaluations,seconds,energy,gradient_norm,precond_reg")
        start = time_ns()
        function record(x, e, g, it)
            seconds = (time_ns()-start)/1e9
            laststate[] = x; iteration[] = it
            x.e0 = e; x.grad = Matrix(g)
            it > 0 && push!(accepted_times, seconds)
            println(io, join((it,calls[],seconds,e,sqrt(metric(x,g,g)),
                             x.manifold_param.precond_reg), ',')); flush(io)
            @printf("BLAS=%d it=%d fg=%d t=%.2fs E=%.14f\n",threads,it,calls[],seconds,e)
            flush(stdout)
        end
        function fg(x)
            calls[] += 1
            result = @timed objective(x)
            push!(fg_times, result.time)
            if firstcall[]
                firstcall[] = false
                record(x,result.value...,0)
            end
            result.value
        end
        function callback(x,e,g,it)
            finalize!(x,e,g,it)
            record(x,e,g,it)
            x,e,g
        end
        OptimKit.optimize(fg,s,
            OptimKit.LBFGS(100;maxiter=3,gradtol=3e-6,verbosity=0,linesearch=ls_default);
            inner=metric,retract=retraction,precondition=precondition,
            transport! = naive_transport!,finalize! = callback,isometrictransport=false,
            hasconverged=(args...)->false)
        elapsed = (time_ns()-start)/1e9
    end
    iteration[] == 3 || error("Did not complete three iterations")
    final = laststate[]
    rcmps_store(final,joinpath(output,tag * "_final.jld2");verbose=false)
    # Last two accepted steps exclude the initial objective and first step setup.
    step_seconds = diff(accepted_times)
    row = (tag,threads,Threads.nthreads(),iteration[],calls[],elapsed,
           sum(fg_times),median(fg_times),median(step_seconds),final.e0,
           sqrt(metric(final,final.grad,final.grad)))
    open(joinpath(output,"summary.csv"),"a") do io
        println(io,join(row,',')); flush(io)
    end
    @printf("DONE %s total=%.3fs median_fg=%.3fs\n",tag,elapsed,median(fg_times)); flush(stdout)
end

function main()
    output = abspath(get(ENV,"RCMPS_ATLAS_OUTPUT",joinpath(@__DIR__,"atlas_blas")))
    mkpath(output)
    isfile(joinpath(output,"summary.csv")) && error("Choose a fresh output directory")
    counts = parse.(Int,split(get(ENV,"RCMPS_BLAS_COUNTS","1,2,4,8,16,24,1"),','))
    BLAS.set_num_threads(1)
    println("CPU=",Sys.cpu_info()[1].model," Julia=",VERSION," Julia threads=",Threads.nthreads())
    println(BLAS.get_config()); flush(stdout)
    initial = rcmps_load(joinpath(@__DIR__,"d64_50","final.jld2"))
    println("Fixed initial energy=",initial.e0," D=",initial.D)
    warm = RandomState(2;seed=6170,scaling=.3)
    warm.model_param.name="phi4"; warm.model_param.coupling=2.
    optimize_state(warm;n_iter=2,verbosity=0,mlbfgs=100,parallel=:both,
                   kinetic_backend=:krylov,kinetic_options=angular,krylov_options=krylov)
    open(joinpath(output,"summary.csv"),"w") do io
        println(io,"tag,blas_threads,julia_threads,iterations,evaluations,optimization_seconds,fg_seconds,median_fg_seconds,median_last_two_step_seconds,energy,gradient_norm")
    end
    for (i,nt) in enumerate(counts)
        run_case(initial,nt,"run$(i)_blas$(nt)",output)
    end
end
main()
