# Additional polishing when the legacy objective's numerical noise blocks strict descent.
include("converged_optimization.jl")
struct PrecisionLineSearch <: OptimKit.AbstractLineSearch
    energy_atol::Float64
end
function (ls::PrecisionLineSearch)(fg,x,p,fg0=fg(x);retract=retraction,inner=metric,
                                  initialguess=1.,acceptfirst=false,kw...)
    cap = 0.5max(1,norm(x.R))/max(norm(p),eps())
    bounded(x,p,a) = abs(a) <= cap ? retract(x,p,a) : throw(SolverFailure("trial step too large"))
    nfg = Ref(0)
    counted(x) = (nfg[]+=1; fg(x))
    try
        result = OptimKit.HagerZhangLineSearch(ϵ=ls.energy_atol)(
            counted,x,p,fg0;retract=bounded,inner,initialguess=min(initialguess,cap),acceptfirst)
        y,f,g,v,a,_ = result
        f <= fg0[1]+ls.energy_atol && a > 0 || throw(SolverFailure("Wolfe trial outside numerical tolerance"))
        return y,f,g,v,a,nfg[]
    catch e
        e isa SolverFailure || rethrow()
    end
    slope = inner(x,fg0[2],p)
    slope < 0 || throw(SolverFailure("not a descent direction"))
    a = min(initialguess,cap)
    for _ in 1:40
        try
            y,v = bounded(x,p,a)
            f,g = counted(y)
            if f <= fg0[1]+1e-4*a*slope+ls.energy_atol
                return y,f,g,v,a,nfg[]
            end
        catch e
            e isa SolverFailure || rethrow()
        end
        a /= 2
    end
    throw(SolverFailure("precision-aware line search failed"))
end
function polish(d,name;maxiter=300,energy_atol=1e-9,gradtol=5e-6)
    old = name == "old"
    mod = old ? Legacy : RcmpsCore
    path = joinpath(output_dir,"D$(d)_$(name).jld2")
    cp = RcmpsCore.JLD2.load(path)
    s = old ? Legacy.State(cp["R"],cp["K"]) : State(cp["R"],cp["K"])
    s.manifold_param.metric_reg = 1e-3
    s.manifold_param.precond_reg_prefactor = 1e-2
    settings!(s,:fine)
    s.integrator_param.atol = 1e-14
    s.integrator_param.rtol = 1e-13
    s.integrator_param.n_quad = old ? 3000 : 256
    nfg,nit = Ref(0),Ref(0)
    laststate = Ref(s)
    start = time_ns()
    fg(x) = (nfg[]+=1; mod.fg_phi4(x,2.))
    status = "iteration_limit"
    histpath = joinpath(output_dir,"D$(d)_$(name)_polish.csv")
    open(histpath,"w") do io
        println(io,"iteration,evaluations,seconds,energy,optimizer_gradient_norm")
        finish(x,f,g,k) = begin
            laststate[]=x;nit[]=k
            gn = sqrt(mod.metric(x,g,g))
            println(io,join((k,nfg[],(time_ns()-start)/1e9,f,gn),','));flush(io)
            if k%10==0 || gn<gradtol
                @printf("POLISH D=%d %s it=%d E=%.12f |g|=%.3e\n",d,name,k,f,gn);flush(stdout)
            end
            mod.finalize!(x,f,g,k)
        end
        try
            x,f,g,_,_ = OptimKit.optimize(fg,s,
                OptimKit.LBFGS(30;maxiter,gradtol,verbosity=0,linesearch=PrecisionLineSearch(energy_atol));
                inner=mod.metric,retract=mod.retraction,precondition=mod.precondition,
                transport! = mod.naive_transport!,finalize! = finish,isometrictransport=false)
            laststate[]=x
            status = sqrt(mod.metric(x,g,g)) <= gradtol ? "optimizer_converged" : "iteration_limit"
        catch e
            e isa InterruptException && rethrow()
            status = "failed: "*replace(sprint(showerror,e),','=>';','\n'=>' ')
        end
    end
    elapsed = (time_ns()-start)/1e9
    s = laststate[]
    ref = State(s.R,s.K)
    ref.manifold_param.metric_reg = 1e-3
    ref.integrator_param.atol=1e-14
    ref.integrator_param.rtol=1e-13
    ref.integrator_param.n_quad=256
    e20,g20 = fg_phi4(ref,2.)
    ref.integrator_param.max_point=30.
    e30,g30 = fg_phi4(ref,2.)
    gn = sqrt(metric(ref,g30,g30))
    ok = gn<1e-5 && abs(e30-e20)<1e-8
    it,n,t = cp["iteration"]+nit[],cp["evaluations"]+nfg[],cp["seconds"]+elapsed
    checkpoint(path,s,it,n,t,e30,gn,:polish)
    row=(d,name,it,n,t,e30,gn,abs(e30-e20),mass_gap(ref),ok,status)
    open(joinpath(output_dir,"summary.csv"),"a") do io
        println(io,join(row,','))
    end
    println("POLISHED ",row);flush(stdout)
end
if abspath(PROGRAM_FILE)==abspath(@__FILE__)
    length(ARGS)==2 || error("usage: polish_converged.jl D old|new")
    polish(parse(Int,ARGS[1]),ARGS[2])
end
