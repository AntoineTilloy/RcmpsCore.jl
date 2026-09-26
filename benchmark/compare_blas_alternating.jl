using MKL # Intel OpenMP must load first.
using RcmpsCore, LinearAlgebra, Printf

function select_backend(name)
    if name == "mkl"
        MKL.lbt_forward_to_mkl()
        MKL.set_num_threads(1)
    else
        BLAS.lbt_forward(LinearAlgebra.OpenBLAS_jll.libopenblas; clear=true)
    end
    BLAS.set_num_threads(1)
end

function main()
    output = joinpath(@__DIR__, "blas_comparison")
    MKL.set_num_threads(1)
    BLAS.set_num_threads(1)
    s = rcmps_load(joinpath(@__DIR__, "d64_50", "final.jld2"))
    evaluate() = fg_phi4(s, 2.; parallel=:both, kinetic_backend=:krylov,
                        kinetic_options=KineticOptions(maxnodes=1024),
                        krylov_options=KineticKrylovOptions(maxdim=1024))
    for name in ("openblas", "mkl")
        select_backend(name)
        println("Warmup ", name, ' ', BLAS.get_config()); flush(stdout)
        evaluate()
    end
    reference = nothing
    open(joinpath(output, "alternating.csv"), "w") do io
        println(io, "sample,backend,seconds,gc_seconds,energy,gradient_norm,energy_difference,gradient_matrix_difference")
        for (i, name) in enumerate(("openblas", "mkl", "mkl", "openblas",
                                    "openblas", "mkl", "mkl", "openblas"))
            # Prior evaluation has joined all tasks before changing process-wide BLAS.
            select_backend(name)
            GC.gc()
            result = @timed evaluate()
            e, g = result.value
            isnothing(reference) && (reference = (e,copy(g)))
            println(io, join((i,name,result.time,result.gctime,e,sqrt(metric(s,g,g)),
                             abs(e-reference[1]),norm(g-reference[2])), ',')); flush(io)
            @printf("%s sample %d %.3fs E=%.14f\n", name,i,result.time,e); flush(stdout)
        end
    end
end
main()
