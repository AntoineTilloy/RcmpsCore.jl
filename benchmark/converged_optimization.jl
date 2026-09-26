# Converged end-to-end g=2 benchmark. Run sequentially, with checkpoints.
include("full_optimization.jl")
const output_dir = joinpath(@__DIR__,"converged")
mkpath(output_dir)

function settings!(s,phase)
    p = s.integrator_param
    p.max_point = 20.
    p.atol = phase == :coarse ? 1e-11 : 1e-13
    p.rtol = phase == :coarse ? 1e-10 : 1e-12
    p.n_quad = s isa State ? (phase == :coarse ? 128 : 192) : (phase == :coarse ? 1000 : 2000)
    return s
end

function checkpoint(path,x,iteration,nfg,seconds,energy,gn,phase)
    tmp = path*".tmp"
    RcmpsCore.JLD2.jldopen(tmp,"w") do f
        f["R"],f["K"] = x.R,x.K
        f["iteration"],f["evaluations"],f["seconds"] = iteration,nfg,seconds
        f["energy"],f["gradient_norm"],f["phase"] = energy,gn,String(phase)
    end
    mv(tmp,path;force=true)
end

function long_run(d,old;maxiter=1000)
    mod = old ? Legacy : RcmpsCore
    name = old ? "old" : "new"
    path = joinpath(output_dir,"D$(d)_$(name).jld2")
    histpath = joinpath(output_dir,"D$(d)_$(name).csv")
    s = initial_state(d,old)
    totalit,totalfg,totaltime = 0,0,0.
    if get(ENV,"RCMPS_RESUME","false") == "true" && isfile(path)
        cp = RcmpsCore.JLD2.load(path)
        s = old ? Legacy.State(cp["R"],cp["K"]) : State(cp["R"],cp["K"])
        s.manifold_param.metric_reg = 1e-3
        totalit,totalfg,totaltime = cp["iteration"],cp["evaluations"],cp["seconds"]
    end
    status = "iteration_limit"
    open(histpath,totalit > 0 ? "a" : "w") do io
        totalit == 0 && println(io,"phase,iteration,evaluations,seconds,energy,optimizer_gradient_norm")
        for phase in (:coarse,:fine)
            settings!(s,phase)
            threshold = phase == :coarse ? 3e-4 : 3e-6
            nfg,nit = Ref(0),Ref(0)
            laststate = Ref(s)
            overhead = Ref(0.)
            start = time_ns()
            elapsed() = (time_ns()-start)/1e9-overhead[]
            fg(x) = (nfg[] += 1; mod.fg_phi4(x,2.))
            finish(x,f,g,k) = begin
                laststate[] = x
                nit[] = k
                gn = sqrt(mod.metric(x,g,g))
                println(io,join((phase,totalit+k,totalfg+nfg[],totaltime+elapsed(),f,gn),',')); flush(io)
                if k % 10 == 0 || gn <= threshold
                    t0 = time_ns()
                    accepted_time = totaltime+elapsed()
                    checkpoint(path,x,totalit+k,totalfg+nfg[],accepted_time,f,gn,phase)
                    @printf("D=%d %s %s iter=%d fg=%d t=%.1fs E=%.12f |g|=%.3e\n",
                            d,name,phase,totalit+k,totalfg+nfg[],accepted_time,f,gn)
                    flush(stdout)
                    overhead[] += (time_ns()-t0)/1e9
                end
                mod.finalize!(x,f,g,k)
            end
            try
                x,f,g,_,history = OptimKit.optimize(fg,s,
                    OptimKit.LBFGS(30;maxiter,gradtol=threshold,verbosity=0,linesearch=RcmpsCore.ls_default);
                    inner=mod.metric,retract=mod.retraction,precondition=mod.precondition,
                    transport! = mod.naive_transport!,finalize! = finish,isometrictransport=false)
                laststate[] = x
                status = sqrt(mod.metric(x,g,g)) <= threshold ? "optimizer_converged" : "iteration_limit"
            catch e
                e isa InterruptException && rethrow()
                status = "failed: "*replace(sprint(showerror,e),','=>';','\n'=>' ')
                println("D=$d $name $phase $status"); flush(stdout)
            end
            totaltime += elapsed()
            totalfg += nfg[]
            totalit += nit[]
            s = laststate[]
            # Save accepted parameters even if a trial failed.
            f,g = mod.fg_phi4(s,2.)
            checkpoint(path,s,totalit,totalfg,totaltime,f,sqrt(mod.metric(s,g,g)),phase)
        end
    end
    ref = State(s.R,s.K)
    ref.manifold_param.metric_reg = 1e-3
    settings!(ref,:fine)
    e20,g20 = fg_phi4(ref,2.)
    ref.integrator_param.max_point = 30.
    ref.integrator_param.n_quad = 256
    e30,g30 = fg_phi4(ref,2.)
    gn = sqrt(metric(ref,g30,g30))
    converged = gn <= 1e-5 && abs(e30-e20) < 1e-8
    @printf("FINAL D=%d %s E=%.12f |g_true|=%.3e window_error=%.2e it=%d t=%.1fs converged=%s\n",
            d,name,e30,gn,abs(e30-e20),totalit,totaltime,converged); flush(stdout)
    return (d,name,totalit,totalfg,totaltime,e30,gn,abs(e30-e20),mass_gap(ref),converged,status)
end

function converged_main()
    dims = parse.(Int,split(get(ENV,"RCMPS_BENCH_DIMS","2,4,8,16"),','))
    implementations = split(get(ENV,"RCMPS_BENCH_IMPLEMENTATIONS","new,old"),',')
    maxiter = parse(Int,get(ENV,"RCMPS_BENCH_STEPS","1000"))
    println("Warmup..."); flush(stdout)
    for name in implementations
        old = name == "old"
        optimize_pipeline(initial_state(2,old),old ? Legacy : RcmpsCore,1)
    end
    summary = joinpath(output_dir,"summary.csv")
    open(summary,"a") do io
        position(io) == 0 && println(io,"D,implementation,iterations,evaluations,seconds,energy,corrected_gradient_norm,window_error,gap,converged,optimizer_status")
        for d in dims, name in implementations
            println("START D=$d $name"); flush(stdout)
            row = long_run(d,name=="old";maxiter)
            println(io,join(row,',')); flush(io)
        end
    end
end
abspath(PROGRAM_FILE) == abspath(@__FILE__) && converged_main()
