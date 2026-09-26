# Load MKL first in its own process; the default process retains OpenBLAS.
const backend = get(ENV, "RCMPS_BLAS", "openblas")
backend in ("openblas", "mkl") || error("Unknown RCMPS_BLAS=$backend")
if backend == "mkl"
    using MKL
    MKL.set_num_threads(1) # Set LAPACK as well as BLAS.
end
using RcmpsCore, LinearAlgebra, Random, Statistics, Printf

function main()
    BLAS.set_num_threads(1)
    output = joinpath(@__DIR__, "blas_comparison")
    mkpath(output)
    println("Julia ", VERSION, "; CPU ", Sys.CPU_NAME, "; threads ", Threads.nthreads())
    println(BLAS.get_config()); flush(stdout)
    rng = MersenneTwister(771)
    open(joinpath(output, backend * "_gemm.csv"), "w") do io
        println(io, "D,median_microseconds")
        for d in (32, 64, 128, 256)
            a = randn(rng, ComplexF64, d, d)
            b = randn(rng, ComplexF64, d, d)
            c = similar(a)
            mul!(c, a, b)
            times = Float64[]
            for repeat in 1:5
                elapsed = @elapsed begin
                    for _ in 1:1000
                        mul!(c, a, b)
                    end
                end
                push!(times, elapsed/1000)
            end
            println(io, d, ',', median(times)*1e6)
        end
    end
    s = rcmps_load(joinpath(@__DIR__, "d64_50", "final.jld2"))
    evaluate() = fg_phi4(s, 2.; parallel=:both, kinetic_backend=:krylov,
                        kinetic_options=KineticOptions(maxnodes=1024),
                        krylov_options=KineticKrylovOptions(maxdim=1024))
    println("Untimed full evaluation warmup"); flush(stdout)
    evaluate()
    open(joinpath(output, backend * "_full.csv"), "w") do io
        println(io, "sample,seconds,gc_seconds,bytes,energy,gradient_norm")
        for sample in 1:3
            GC.gc()
            result = @timed evaluate()
            e, g = result.value
            println(io, join((sample,result.time,result.gctime,result.bytes,e,sqrt(metric(s,g,g))), ','))
            flush(io)
            @printf("%s sample %d: %.3fs E=%.14f\n", backend, sample, result.time, e)
            flush(stdout)
            s.e0 = e; s.grad = Matrix(g)
        end
    end
    rcmps_store(s, joinpath(output, backend * "_result.jld2"); verbose=false)
    if backend == "mkl" && isfile(joinpath(output, "openblas_result.jld2"))
        reference = rcmps_load(joinpath(output, "openblas_result.jld2"))
        open(joinpath(output, "agreement.csv"), "w") do io
            println(io, "energy_difference,gradient_matrix_difference,relative_gradient_difference")
            println(io, join((abs(s.e0-reference.e0),norm(s.grad-reference.grad),
                             norm(s.grad-reference.grad)/norm(reference.grad)), ','))
        end
    end
end
abspath(PROGRAM_FILE) == abspath(@__FILE__) && main()
