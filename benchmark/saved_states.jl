using RcmpsCore, LinearAlgebra, Random, Printf, JLD
BLAS.set_num_threads(1)
function main()
    rows = ["file,D,gap,saved_energy,new_energy,bulk_error,corrected_error,boundary_norm,seconds"]
    for file in ("phi4_g1.0_D8_Hrjz1lQv4R.jld","phi4_g2.0_D16_.jld","phi4_g2.0_D28_BaBajp39uG.jld")
        data = JLD.load(joinpath(@__DIR__,"legacy_states",file))
        s = State(data["R"],data["K"])
        s.integrator_param.atol = 1e-12
        s.integrator_param.rtol = 1e-12
        s.integrator_param.n_quad = 192
        g = real(data["coupling"])
        w = randn(MersenneTwister(90),ComplexF64,s.D,s.D); w /= norm(w)
        gap = mass_gap(s;method=:dense)
        t = @elapsed value,grad = fg_phi4(s,g)
        _,bulk = fg_phi4(s,g;boundary=false)
        h = 1e-4
        p,_ = retraction(s,w,h); m,_ = retraction(s,w,-h)
        fd = (fg_phi4(p,g;gradient=false)-fg_phi4(m,g;gradient=false))/(2h)
        e1,e2 = abs(metric(s,bulk,w)-fd),abs(metric(s,grad,w)-fd)
        @printf("%s: gap %.4g, bulk error %.3e, corrected %.3e, E %.10g\n",file,gap,e1,e2,value)
        push!(rows,join((file,s.D,gap,data["e0"],value,e1,e2,norm(grad-bulk),t),','))
    end
    write(joinpath(@__DIR__,"saved_states.csv"),join(rows,'\n')*"\n")
end
main()
