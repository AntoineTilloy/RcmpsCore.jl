# Sequential D32 phi4 optimization, g=2 followed by continuation to g=1.
# Run with six Julia threads, one GC thread. BLAS is explicitly single threaded.
module OptimizeD32
using RcmpsCore, LinearAlgebra, Random, Printf, Dates
import OptimKit
const RC=RcmpsCore
const angular=KineticOptions(maxnodes=1024)
const krylov=KineticKrylovOptions(maxdim=1024)
const output=abspath(get(ENV,"RCMPS_D32_OUTPUT",joinpath(@__DIR__,"d32_optimization")))

mutable struct Monitor
    io::IO
    coupling::Float64
    phase::Symbol
    attempt::Int
    start_ns::UInt64
    iteration_offset::Int
    phase_iterations::Int
    evaluations::Int
    laststate::State
    last_gradient_norm::Float64
end
elapsed(m)=Float64(time_ns()-m.start_ns)/1e9
label(g)="g$(Int(g))"
function record!(m,s,energy,gradient,iteration)
    m.laststate=s;m.phase_iterations=iteration
    m.last_gradient_norm=sqrt(metric(s,gradient,gradient))
    s.e0=energy;s.grad=Matrix(gradient)
    total=m.iteration_offset+iteration
    println(m.io,join((m.coupling,m.phase,m.attempt,total,m.evaluations,elapsed(m),energy,m.last_gradient_norm),','))
    flush(m.io)
    @printf("g=%.1f %-6s it=%d fg=%d t=%.1fs E=%.14f |g|=%.4e\n",
            m.coupling,m.phase,total,m.evaluations,elapsed(m),energy,m.last_gradient_norm)
    flush(stdout)
    if iteration%10==0
        rcmps_store(s,joinpath(output,label(m.coupling)*"_checkpoint.jld2");verbose=false)
    end
end
function phase!(m,s,phase)
    fine=phase==:fine
    s.integrator_param.atol=fine ? 1e-13 : 1e-11
    s.integrator_param.rtol=fine ? 1e-12 : 1e-10
    s.integrator_param.n_quad=fine ? 192 : 128
    target=fine ? 3e-6 : 3e-4
    m.phase=phase
    for attempt in 1:3
        m.attempt=attempt;m.phase_iterations=0
        first_call=Ref(true)
        function objective(x)
            m.evaluations+=1
            result=fg_phi4(x,m.coupling;parallel=:both,kinetic_backend=:krylov,
                           kinetic_options=angular,krylov_options=krylov)
            if first_call[]
                first_call[]=false
                record!(m,x,result[1],result[2],0)
            end
            result
        end
        function callback(x,value,gradient,iteration)
            finalize!(x,value,gradient,iteration)
            record!(m,x,value,gradient,iteration)
            x,value,gradient
        end
        try
            result=OptimKit.optimize(objective,s,
                OptimKit.LBFGS(30;maxiter=2000,gradtol=target,verbosity=0,linesearch=ls_default);
                inner=metric,retract=retraction,precondition=precondition,
                transport! = naive_transport!,finalize! = callback,isometrictransport=false)
            m.laststate=result[1]
            m.last_gradient_norm=sqrt(metric(result[1],result[3],result[3]))
        catch failure
            failure isa InterruptException && rethrow()
            failure isa SolverFailure || rethrow()
            println("RECOVERY ",phase," attempt=",attempt," ",sprint(showerror,failure));flush(stdout)
        end
        s=m.laststate
        m.iteration_offset+=m.phase_iterations
        rcmps_store(s,joinpath(output,label(m.coupling)*"_checkpoint.jld2");verbose=false)
        m.last_gradient_norm<=target && return s
        println("Restarting LBFGS from last accepted state; |g|=",m.last_gradient_norm);flush(stdout)
    end
    error("$(m.phase) did not converge; checkpoint retained")
end

function verify(s,coupling,iterations,evaluations,seconds)
    println("VERIFY g=",coupling," independent ODE L20/L30");flush(stdout)
    reference=State(s.R,s.K);copy_param!(reference,s)
    reference.integrator_param.atol=1e-13
    reference.integrator_param.rtol=1e-12
    reference.integrator_param.n_quad=256
    reference.integrator_param.max_point=20.
    energy20,_=fg_phi4(reference,coupling;parallel=:both)
    reference.integrator_param.max_point=30.
    energy30,gradient30=fg_phi4(reference,coupling;parallel=:both)
    norm30=sqrt(metric(reference,gradient30,gradient30))
    window_error=abs(energy30-energy20)
    converged=norm30<1e-5 && window_error<1e-8
    open(joinpath(output,"summary.csv"),"a") do io
        position(io)==0 && println(io,"coupling,D,iterations,evaluations,optimization_seconds,energy,verified_gradient_norm,window_error,converged")
        println(io,join((coupling,s.D,iterations,evaluations,seconds,energy30,norm30,window_error,converged),','))
    end
    @printf("FINAL g=%.1f D=%d it=%d fg=%d t=%.1fs E=%.14f verified |g|=%.4e window_error=%.3e converged=%s\n",
            coupling,s.D,iterations,evaluations,seconds,energy30,norm30,window_error,string(converged))
    flush(stdout)
    s.e0=energy30;s.grad=gradient30
    s.info="D32 phi4 g=$coupling; independent ODE L30 verification; converged=$converged"
    rcmps_store(s,joinpath(output,label(coupling)*"_final.jld2");verbose=false)
    converged || error("Independent convergence check failed; final state retained")
    s
end
function run(s,coupling)
    s.model_param.name="phi4";s.model_param.coupling=coupling
    s.manifold_param.precond_reg=1e-4
    println("START g=",coupling," at ",Dates.now());flush(stdout)
    open(joinpath(output,label(coupling)*"_trajectory.csv"),"w") do io
        println(io,"coupling,phase,attempt,iteration,evaluations,seconds,energy,gradient_norm")
        monitor=Monitor(io,coupling,:coarse,1,time_ns(),0,0,0,s,Inf)
        state=phase!(monitor,s,:coarse)
        state=phase!(monitor,state,:fine)
        runtime=elapsed(monitor)
        rcmps_store(state,joinpath(output,label(coupling)*"_optimized.jld2");verbose=false)
        verify(state,coupling,monitor.iteration_offset,monitor.evaluations,runtime)
    end
end
function main()
    mkpath(output)
    isfile(joinpath(output,"g2_trajectory.csv")) && error("Refusing to overwrite an existing run; select a fresh RCMPS_D32_OUTPUT")
    BLAS.set_num_threads(1)
    println("Julia ",VERSION,"; compute threads=",Threads.nthreads(),"; BLAS=1; output=",output);flush(stdout)
    warm=RandomState(2;seed=6132,scaling=.3)
    warm.model_param.name="phi4";warm.model_param.coupling=2.
    optimize_state(warm;n_iter=2,verbosity=0,parallel=:both,kinetic_backend=:krylov,
                   kinetic_options=angular,krylov_options=krylov)
    rng=MersenneTwister(6132)
    r=randn(rng,ComplexF64,32,32)/sqrt(32)
    k=randn(rng,ComplexF64,32,32)/sqrt(32)
    s=State(r,k+k')
    s.model_param.name="phi4";s.model_param.coupling=2.
    s.manifold_param.metric_reg=1e-3
    s.manifold_param.precond_reg_prefactor=1e-2
    s.integrator_param.max_point=20.
    rcmps_store(s,joinpath(output,"initial.jld2");verbose=false)
    g2=run(s,2.)
    run(deepcopy(g2),1.)
    println("Both couplings converged and independently verified.");flush(stdout)
end
end
abspath(PROGRAM_FILE)==abspath(@__FILE__) && OptimizeD32.main()
