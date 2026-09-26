# One uninterrupted 50-iteration D64 phi4 run at fixed fine precision.
module OptimizeD64Test
using RcmpsCore, LinearAlgebra, Random, Printf
import OptimKit
const output=abspath(get(ENV,"RCMPS_D64_OUTPUT",joinpath(@__DIR__,"d64_50")))
const angular=KineticOptions(maxnodes=1024)
const krylov=KineticKrylovOptions(maxdim=1024)

function main()
    mkpath(output)
    trajectory=joinpath(output,"trajectory.csv")
    isfile(trajectory) && error("Refusing to overwrite an existing trajectory; choose a fresh RCMPS_D64_OUTPUT")
    BLAS.set_num_threads(1)
    println("D64 g=2, fixed fine precision, 50 iterations, LBFGS history=100; Julia ",VERSION,
            "; compute threads=",Threads.nthreads(),"; BLAS=1");flush(stdout)
    warm=RandomState(2;seed=6164,scaling=.3)
    warm.model_param.name="phi4";warm.model_param.coupling=2.
    optimize_state(warm;n_iter=2,verbosity=0,mlbfgs=100,parallel=:both,
                   kinetic_backend=:krylov,kinetic_options=angular,krylov_options=krylov)

    rng=MersenneTwister(6164)
    r=randn(rng,ComplexF64,64,64)/sqrt(64)
    k=randn(rng,ComplexF64,64,64)/sqrt(64)
    initial=State(r,k+k')
    initial.model_param.name="phi4";initial.model_param.coupling=2.
    initial.manifold_param.metric_reg=1e-3
    initial.manifold_param.precond_reg=1e-4
    initial.manifold_param.precond_reg_prefactor=1e-2
    initial.integrator_param.max_point=20.
    initial.integrator_param.atol=1e-13
    initial.integrator_param.rtol=1e-12
    initial.integrator_param.n_quad=192
    initial.info="D64 g=2 random initial state; seed 6164; fixed precision 50-iteration test"
    rcmps_store(initial,joinpath(output,"initial.jld2");verbose=false)
    laststate=Ref(initial);iterations=Ref(0);evaluations=Ref(0);first_call=Ref(true)
    start=time_ns()
    open(trajectory,"w") do io
        println(io,"iteration,evaluations,seconds,energy,gradient_norm,metric_reg,precond_reg")
        function record(x,value,gradient,iteration)
            laststate[]=x;iterations[]=iteration
            x.e0=value;x.grad=Matrix(gradient)
            elapsed=Float64(time_ns()-start)/1e9
            gn=sqrt(metric(x,gradient,gradient))
            println(io,join((iteration,evaluations[],elapsed,value,gn,
                             x.manifold_param.metric_reg,x.manifold_param.precond_reg),','));flush(io)
            @printf("it=%d fg=%d t=%.1fs E=%.14f |g|=%.4e precond_reg=%.3e\n",
                    iteration,evaluations[],elapsed,value,gn,x.manifold_param.precond_reg);flush(stdout)
            # Small atomic checkpoints retain every accepted state, including on interruption.
            rcmps_store(x,joinpath(output,"checkpoint.jld2");verbose=false)
        end
        function objective(x)
            evaluations[]+=1
            result=fg_phi4(x,2.;parallel=:both,kinetic_backend=:krylov,
                           kinetic_options=angular,krylov_options=krylov)
            if first_call[]
                first_call[]=false
                record(x,result[1],result[2],0)
            end
            result
        end
        function callback(x,value,gradient,iteration)
            finalize!(x,value,gradient,iteration)
            record(x,value,gradient,iteration)
            x,value,gradient
        end
        OptimKit.optimize(objective,initial,
            OptimKit.LBFGS(100;maxiter=50,gradtol=3e-6,verbosity=0,linesearch=ls_default);
            inner=metric,retract=retraction,precondition=precondition,
            transport! = naive_transport!,finalize! = callback,isometrictransport=false,
            hasconverged=(args...)->false)
    end
    seconds=Float64(time_ns()-start)/1e9
    iterations[]==50 || error("Run did not complete the requested 50 iterations")
    final=laststate[]
    final.info="Completed 50 uninterrupted LBFGS iterations at fixed fine precision; not a convergence claim"
    rcmps_store(final,joinpath(output,"final.jld2");verbose=false)
    gn=sqrt(metric(final,final.grad,final.grad))
    open(joinpath(output,"summary.csv"),"w") do io
        println(io,"D,coupling,iterations,evaluations,optimization_seconds,energy,gradient_norm,lbfgs_history")
        println(io,join((64,2.,iterations[],evaluations[],seconds,final.e0,gn,100),','))
    end
    @printf("FINAL 50 iterations: E=%.14f |g|=%.4e seconds=%.1f evaluations=%d\n",
            final.e0,gn,seconds,evaluations[]);flush(stdout)

    println("Independent final ODE check, L30 and 256 nodes per half (outside optimization timer)");flush(stdout)
    reference=deepcopy(final)
    reference.integrator_param.max_point=30.
    reference.integrator_param.n_quad=256
    er,gr=fg_phi4(reference,2.;parallel=:both,kinetic_backend=:ode)
    open(joinpath(output,"verification.csv"),"w") do io
        println(io,"ode_energy,ode_gradient_norm,energy_difference,gradient_matrix_difference")
        println(io,join((er,sqrt(metric(reference,gr,gr)),abs(er-final.e0),norm(gr-final.grad)),','))
    end
    @printf("VERIFIED reference E=%.14f energy difference=%.3e gradient difference=%.3e\n",
            er,abs(er-final.e0),norm(gr-final.grad));flush(stdout)
end
end
abspath(PROGRAM_FILE)==abspath(@__FILE__) && OptimizeD64Test.main()
