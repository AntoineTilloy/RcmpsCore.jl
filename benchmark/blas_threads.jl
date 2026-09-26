include("threading.jl")
open(joinpath(destination,"blas_threads.csv"),"w") do io
    println(io,"D,state,blas_threads,min_seconds,median_seconds,energy_error,gradient_error")
    for (d,kind) in ((16,"converged"),(32,"initial"))
        BLAS.set_num_threads(1)
        s=benchmark_state(d,kind;fine=kind=="converged")
        e0,g0=fg_phi4(s,2.)
        for nt in (1,2,3,6)
            BLAS.set_num_threads(nt)
            b,med=sample(()->fg_phi4(s,2.;parallel=:serial);count=3)
            e,g=b.value
            row=(d,kind,nt,b.time,med,abs(e-e0),norm(g-g0))
            println(io,join(row,','));flush(io)
            println(row);flush(stdout)
        end
    end
end
BLAS.set_num_threads(1)
