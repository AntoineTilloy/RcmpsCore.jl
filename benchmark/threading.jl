# Run separately with --threads=1,2,3,6 and --gcthreads=1. No concurrent benchmarks.
using RcmpsCore, LinearAlgebra, Random, Printf
using OrdinaryDiffEqVerner: Vern7
const RC = RcmpsCore
const destination = joinpath(@__DIR__,"threading")
mkpath(destination)
BLAS.set_num_threads(parse(Int,get(ENV,"RCMPS_BLAS_THREADS","1")))

function benchmark_state(d,kind;fine=false)
    if kind == "initial"
        rng = MersenneTwister(6100+d)
        r = randn(rng,ComplexF64,d,d)/sqrt(d)
        k = randn(rng,ComplexF64,d,d)/sqrt(d)
        s = State(r,k+k')
    else
        cp = RC.JLD2.load(joinpath(@__DIR__,"converged","D$(d)_new.jld2"))
        s = State(cp["R"],cp["K"])
    end
    s.model_param.name = "phi4"
    s.model_param.coupling = 2.
    s.manifold_param.metric_reg = 1e-3
    s.manifold_param.precond_reg = 1e-4
    s.manifold_param.precond_reg_prefactor = 1e-2
    s.integrator_param.max_point = 20.
    s.integrator_param.atol = fine ? 1e-13 : 1e-11
    s.integrator_param.rtol = fine ? 1e-12 : 1e-10
    s.integrator_param.n_quad = fine ? 192 : 128
    return s
end

function sample(f;count=3)
    f() # compilation and initialization excluded
    measurements = [@timed f() for _ in 1:count]
    best = measurements[argmin(getproperty.(measurements,:time))]
    times = sort(getproperty.(measurements,:time))
    return best,times[cld(count,2)]
end

function component_profile(d,kind;fine=false)
    s = benchmark_state(d,kind;fine)
    grid = RC.quadrature(s.integrator_param)
    points,weights = grid
    path = joinpath(destination,"components.csv")
    open(path,"a") do io
        position(io)==0 && println(io,"D,state,fine,component,seconds,bytes,gc_seconds,rhs_calls,accepted_steps,rejected_steps")
        function record(name,f;ode=false)
            b,_ = sample(f;count=2)
            stats = ode ? b.value.stats : nothing
            ns = ode ? (stats.nf,stats.naccept,stats.nreject) : (0,0,0)
            println(io,join((d,kind,fine,name,b.time,b.bytes,b.gctime,ns...),','));flush(io)
            return b.value
        end
        record("stationary",()->steady_state(s))
        record("quadrature",()->RC.quadrature(s.integrator_param))
        covector,env = zero(s.R),zero(s.R)
        for (label,spec,c) in (("kinetic1",RC.kinetic_spec(false),1.),
                               ("kinetic2",RC.kinetic_spec(true),1.),
                               ("quartic",RC.monomial_spec(4),2.))
            op = spec.commutator ? s.Q*s.R-s.R*s.Q : s.R
            fwd = record(label*"_forward",()->RC.trajectory(s,spec,op,points,false,Vern7());ode=true)
            bwd = record(label*"_backward",()->RC.trajectory(s,spec,op,points,true,Vern7());ode=true)
            v,a = record(label*"_contractions",()->RC.observable_pullback(s,spec,points,weights,fwd,bwd))
            covector .+= c*v; env .+= c*a
        end
        record("boundary_and_metric",()->RC.finish_gradient(s,copy(covector),env))
    end
    println("PROFILE D=$d $kind fine=$fine");flush(stdout)
end

function sweep()
    dims = parse.(Int,split(get(ENV,"RCMPS_BENCH_DIMS","4,8,16"),','))
    count = parse(Int,get(ENV,"RCMPS_SAMPLES","3"))
    nt,nb = Threads.nthreads(),BLAS.get_num_threads()
    path = joinpath(destination,"evaluations_t$(nt)_b$(nb).csv")
    open(path,"w") do io
        println(io,"D,state,fine,julia_threads,blas_threads,mode,min_seconds,median_seconds,bytes,gc_seconds,energy_error,gradient_error")
        for d in dims, kind in ("initial","converged")
            fine = kind == "converged"
            s = benchmark_state(d,kind;fine)
            reference = fg_phi4(s,2.)
            for mode in (:serial,:terms,:trajectories,:both)
                b,med = sample(()->fg_phi4(s,2.;parallel=mode);count)
                e,g = b.value
                de,dg = abs(e-reference[1]),norm(g-reference[2])
                de == 0 && dg == 0 || error("threading changed result: $de, $dg")
                println(io,join((d,kind,fine,nt,nb,mode,b.time,med,b.bytes,b.gctime,de,dg),','));flush(io)
                @printf("D=%d %-9s t=%d b=%d %-12s %.4fs (median %.4fs)\n",d,kind,nt,nb,mode,b.time,med);flush(stdout)
            end
        end
    end
end

if abspath(PROGRAM_FILE)==abspath(@__FILE__)
    if "profile" in ARGS
        for d in (8,16), kind in ("initial","converged")
            component_profile(d,kind;fine=kind=="converged")
        end
    else
        sweep()
    end
end
