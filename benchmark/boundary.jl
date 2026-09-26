using RcmpsCore, LinearAlgebra, Random, Printf
BLAS.set_num_threads(1)

function main()
    rows = String["gap,window,value,bulk_error,corrected_error,boundary_seconds,total_seconds"]
    base = RandomState(3;seed=718)
    w = randn(MersenneTwister(83),ComplexF64,3,3); w /= norm(w)
    for scale in (1.0,0.1,0.01)
        s = State(sqrt(scale)*base.R,scale*base.K)
        s.integrator_param.atol = 1e-12
        s.integrator_param.rtol = 1e-12
        for window in (4.,8.,16.,32.)
            s.integrator_param.max_point = window
            value,full = vertex(s,.4)
            _,bulk = vertex(s,.4;boundary=false)
            h = 1e-5
            p,_ = retraction(s,w,h); m,_ = retraction(s,w,-h)
            fd = real(vertex(p,.4;gradient=false)-vertex(m,.4;gradient=false))/(2h)
            bulkerror = abs(metric(s,bulk,w)-fd)
            fullerror = abs(metric(s,full,w)-fd)
            # Warmed timings; boundary is benchmarked on its actual RHS.
            _,c,a = RcmpsCore.evaluate(s,RcmpsCore.vertex_spec(.4);raw=true)
            boundary_environment(s,a)
            bt = minimum(@elapsed(boundary_environment(s,a)) for _ in 1:3)
            ft = minimum(@elapsed(vertex(s,.4)) for _ in 1:3)
            gap = mass_gap(s)
            push!(rows,join((gap,window,real(value),bulkerror,fullerror,bt,ft),','))
            @printf("gap %.4g L %2.0f: bulk %.3e, corrected %.3e; boundary %.2f%% of evaluation\n",
                    gap,window,bulkerror,fullerror,100bt/ft)
        end
    end
    write(joinpath(@__DIR__,"boundary.csv"),join(rows,'\n')*"\n")
end
main()
