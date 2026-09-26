# Matched-start, matched-accuracy complete optimizations; run one process at a time.
include("threading.jl")
import OptimKit

function full_run(d,parallel;maxiter=1000,kinetic_backend=:ode)
    s = benchmark_state(d,"initial")
    nit,nfg,seconds = 0,0,0.
    label = "D$(d)_$(kinetic_backend)_$(parallel)_t$(Threads.nthreads())"
    status = "iteration_limit"
    open(joinpath(destination,label*".csv"),"w") do io
        println(io,"phase,iteration,evaluations,seconds,energy,gradient_norm")
        for phase in (:coarse,:fine)
            p = s.integrator_param
            p.atol,p.rtol,p.n_quad = phase==:coarse ? (1e-11,1e-10,128) : (1e-13,1e-12,192)
            tol = phase==:coarse ? 3e-4 : 3e-6
            calls,steps,laststate = Ref(0),Ref(0),Ref(s)
            start = time_ns()
            fg(x) = (calls[]+=1; fg_phi4(x,2.;parallel,kinetic_backend))
            finish(x,f,g,k) = begin
                steps[]=k;laststate[]=x
                gn = sqrt(metric(x,g,g))
                elapsed = (time_ns()-start)/1e9
                println(io,join((phase,nit+k,nfg+calls[],seconds+elapsed,f,gn),','));flush(io)
                if k%10==0 || gn<=tol
                    @printf("%s %s it=%d fg=%d t=%.1fs E=%.12f |g|=%.2e\n",
                            label,phase,nit+k,nfg+calls[],seconds+elapsed,f,gn);flush(stdout)
                end
                finalize!(x,f,g,k)
            end
            try
                x,f,g,_,_ = OptimKit.optimize(fg,s,
                    OptimKit.LBFGS(30;maxiter,gradtol=tol,verbosity=0,linesearch=ls_default);
                    inner=metric,retract=retraction,precondition=precondition,
                    transport! = naive_transport!,finalize! = finish,isometrictransport=false)
                laststate[]=x
                status = sqrt(metric(x,g,g))<=tol ? "optimizer_converged" : "iteration_limit"
            catch e
                e isa InterruptException && rethrow()
                status = "failed: "*replace(sprint(showerror,e),','=>';','\n'=>' ')
                println(label," ",status);flush(stdout)
            end
            nit+=steps[];nfg+=calls[];seconds+=(time_ns()-start)/1e9
            s=laststate[]
            RC.JLD2.jldsave(joinpath(destination,label*".jld2");R=s.R,K=s.K,iterations=nit,
                            evaluations=nfg,seconds,phase)
        end
    end
    ref = State(s.R,s.K)
    ref.manifold_param.metric_reg=1e-3
    ref.integrator_param.atol=1e-13;ref.integrator_param.rtol=1e-12
    ref.integrator_param.n_quad=256
    e20,_=fg_phi4(ref,2.)
    ref.integrator_param.max_point=30.
    e,g=fg_phi4(ref,2.)
    gn=sqrt(metric(ref,g,g));de=abs(e-e20)
    ok=gn<1e-5 && de<1e-8
    row=(d,kinetic_backend,parallel,Threads.nthreads(),BLAS.get_num_threads(),nit,nfg,seconds,e,gn,de,ok,status)
    open(joinpath(destination,"optimizations.csv"),"a") do io
        position(io)==0 && println(io,"D,kinetic_backend,mode,julia_threads,blas_threads,iterations,evaluations,seconds,energy,corrected_gradient_norm,window_error,converged,status")
        println(io,join(row,','))
    end
    println("FINAL ",row);flush(stdout)
end

if abspath(PROGRAM_FILE)==abspath(@__FILE__)
    dims=parse.(Int,split(get(ENV,"RCMPS_BENCH_DIMS","4,8,16"),','))
    modes=Symbol.(split(get(ENV,"RCMPS_MODES","serial,both"),','))
    kinetic_backend=Symbol(get(ENV,"RCMPS_KINETIC_BACKEND","ode"))
    for mode in modes
        s=benchmark_state(2,"initial")
        optimize_state(s;n_iter=2,parallel=mode,kinetic_backend,verbosity=0)
    end
    for d in dims, mode in modes
        full_run(d,mode;kinetic_backend)
    end
end
