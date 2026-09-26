# End-to-end g=2 comparison. Run from the repository root with --project=benchmark.
# Both pipelines use current OptimKit, matched settings and canonical retraction.
using RcmpsCore, LinearAlgebra, Random, Printf, OptimKit
include("legacy.jl")
BLAS.set_num_threads(1)

function initial_state(d,old)
    rng = MersenneTwister(6100+d)
    r = randn(rng,ComplexF64,d,d)/sqrt(d)
    k = randn(rng,ComplexF64,d,d)/sqrt(d)
    k += k'
    s = old ? Legacy.State(r,k) : State(r,k)
    s.model_param.name = "phi4"
    s.model_param.coupling = 2.0
    s.integrator_param.atol = 1e-11
    s.integrator_param.rtol = 1e-10
    s.integrator_param.max_point = 20.0
    s.manifold_param.metric_reg = 1e-3
    s.manifold_param.precond_reg = 1e-4
    s.manifold_param.precond_reg_prefactor = 1e-2
    return s
end

function optimize_pipeline(s,mod,steps;history_path=nothing)
    laststate = Ref(s)
    nfg,nit = Ref(0),Ref(0)
    records = Tuple{Int,Int,Float64,Float64,Float64}[]
    start = time_ns()
    fg(x) = begin
        nfg[] += 1
        mod.fg_phi4(x,2.)
    end
    finish(x,f,g,k) = begin
        laststate[] = x
        nit[] = k
        push!(records,(k,nfg[],(time_ns()-start)/1e9,f,sqrt(mod.metric(x,g,g))))
        mod.finalize!(x,f,g,k)
    end
    status = "iteration_limit"
    try
        x,f,g,n,history = OptimKit.optimize(fg,s,
            OptimKit.LBFGS(20;maxiter=steps,gradtol=1e-5,verbosity=0,linesearch=OptimKit.HagerZhangLineSearch());
            inner=mod.metric,retract=mod.retraction,precondition=mod.precondition,
            transport! = mod.naive_transport!,finalize! = finish,isometrictransport=false)
        laststate[] = x
        sqrt(mod.metric(x,g,g)) <= 1e-5 && (status="optimizer_converged")
    catch e
        e isa InterruptException && rethrow()
        status = "failed: "*replace(sprint(showerror,e),','=>';','\n'=>' ')
    end
    elapsed = (time_ns()-start)/1e9
    if !isnothing(history_path)
        open(history_path,"w") do io
            println(io,"iteration,evaluations,seconds,energy,optimizer_gradient_norm")
            for record in records
                println(io,join(record,','))
            end
        end
    end
    return laststate[],elapsed,nfg[],nit[],status
end

function main()
    steps = parse(Int,get(ENV,"RCMPS_BENCH_STEPS","30"))
    dims = parse.(Int,split(get(ENV,"RCMPS_BENCH_DIMS","2,4,8,16"),','))
    println("Warming both complete optimization pipelines (excluded from timings)..."); flush(stdout)
    for (old,mod) in ((true,Legacy),(false,RcmpsCore))
        _,_,_,_,status = optimize_pipeline(initial_state(2,old),mod,1)
        startswith(status,"failed") && error("Warmup failed: $status")
    end
    path = joinpath(@__DIR__,"full_optimization.csv")
    open(path,"w") do io
        println(io,"D,implementation,seconds,iterations,evaluations,initial_energy,verified_final_energy,corrected_gradient_norm,gap,status")
        flush(io)
        for d in dims
            for (old,mod) in ((true,Legacy),(false,RcmpsCore))
                name = old ? "old" : "new"
                s = initial_state(d,old)
                # Same independent evaluator for both initial and final energies.
                reference = State(s.R,s.K)
                initial = fg_phi4(reference,2.;gradient=false)
                println("Starting D=$d $name ($steps iterations maximum)"); flush(stdout)
                hist = joinpath(@__DIR__,"full_optimization_D$(d)_$(name).csv")
                out,t,nfg,nit,status = optimize_pipeline(s,mod,steps;history_path=hist)
                verify = State(out.R,out.K)
                verify.manifold_param.metric_reg = 1e-3
                energy,g = fg_phi4(verify,2.)
                gn = sqrt(metric(verify,g,g))
                gap = mass_gap(verify)
                println(io,join((d,name,t,nit,nfg,initial,energy,gn,gap,status),',')); flush(io)
                @printf("D=%2d %-3s %8.2fs %2d it %3d fg E=% .10f true |g|=%.3e [%s]\n",
                        d,name,t,nit,nfg,energy,gn,status); flush(stdout)
            end
        end
    end
end
abspath(PROGRAM_FILE) == abspath(@__FILE__) && main()
