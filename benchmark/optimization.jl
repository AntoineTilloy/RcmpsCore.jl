using RcmpsCore, LinearAlgebra, Printf, OptimKit
BLAS.set_num_threads(1)
function runopt(s,corrected,steps)
    laststate = Ref(deepcopy(s))
    evaluations,iterations = Ref(0),Ref(0)
    fg(x) = (evaluations[] += 1; fg_phi4(x,2.;boundary=corrected))
    finish(x,f,g,k) = begin
        laststate[] = x
        iterations[] = k
        finalize!(x,f,g,k)
    end
    status = "iteration_limit"
    start = time_ns()
    try
        x,energy,grad,numfg,history = OptimKit.optimize(fg,laststate[],
            OptimKit.LBFGS(8;maxiter=steps,gradtol=1e-7,verbosity=0,linesearch=ls_backward);
            retract=retraction,inner=metric,transport! = naive_transport!,
            precondition=precondition,finalize! = finish,isometrictransport=false)
        laststate[] = x
    catch e
        e isa SolverFailure || rethrow()
        status = replace(e.message,','=>';')
    end
    elapsed = (time_ns()-start)/1e9
    energy,truth = fg_phi4(laststate[],2.)
    gn = sqrt(metric(laststate[],truth,truth))
    gn <= 1e-7 && (status="converged")
    return iterations[],evaluations[],energy,gn,elapsed,status
end
function main()
    rows = ["boundary,iterations,evaluations,initial_energy,final_energy,true_gradient_norm,seconds,status"]
    base = RandomState(3;seed=718)
    s = State(.1base.R,.01base.K)
    s.integrator_param.max_point = 12
    s.model_param.name = "phi4"
    s.model_param.coupling = 2
    initial = fg_phi4(s,2.;gradient=false)
    for corrected in (false,true)
        runopt(s,corrected,1)
        it,nfg,energy,gn,elapsed,status = runopt(s,corrected,20)
        @printf("boundary=%s E %.8g -> %.8g, true gradient %.3e, %.2fs; %s\n",corrected,initial,energy,gn,elapsed,status)
        push!(rows,join((corrected,it,nfg,initial,energy,gn,elapsed,status),','))
    end
    write(joinpath(@__DIR__,"optimization.csv"),join(rows,'\n')*"\n")
end
main()
