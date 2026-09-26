include("threading.jl")
path=joinpath(destination,"kinetic_backends_t$(Threads.nthreads())_b$(BLAS.get_num_threads()).csv")
open(path,"w") do io
    println(io,"D,state,backend,mode,min_seconds,median_seconds,bytes,gc_seconds,energy_difference,gradient_difference")
    for d in (4,8,16), kind in ("initial","converged")
        s=benchmark_state(d,kind;fine=kind=="converged")
        e0,g0=fg_phi4(s,2.)
        for backend in (:ode,:resolvent), mode in (:serial,:terms,:both)
            b,med=sample(()->fg_phi4(s,2.;parallel=mode,kinetic_backend=backend);count=3)
            e,g=b.value
            row=(d,kind,backend,mode,b.time,med,b.bytes,b.gctime,abs(e-e0),norm(g-g0))
            println(io,join(row,','));flush(io)
            println(row);flush(stdout)
        end
    end
end
