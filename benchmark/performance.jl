using RcmpsCore, LinearAlgebra, Random, Printf, JLD
include("legacy.jl")
BLAS.set_num_threads(1)

function measure(f;samples=3)
    f()
    stats = [@timed f() for _ in 1:samples]
    return minimum(x.time for x in stats),minimum(x.bytes for x in stats)
end
function main()
    rows = ["kind,D,old_seconds,new_seconds,speedup,old_bytes,new_bytes,value_error,gradient_difference"]
    for d in (4,8,16,32)
        s = RandomState(d;seed=61,scaling=1/sqrt(d))
        old = Legacy.State(s.R,s.K;ρ0=s.ρ0)
        # Same ODE tolerances and physical window, each implementation's default quadrature.
        old.integrator_param.atol = s.integrator_param.atol
        old.integrator_param.rtol = s.integrator_param.rtol
        old.manifold_param.metric_reg = s.manifold_param.metric_reg
        a,ga = Legacy.phi_monomial(old,4)
        b,gb = phi_monomial(s,4)
        ot,ob = measure(()->Legacy.phi_monomial(old,4))
        nt,nb = measure(()->phi_monomial(s,4))
        @printf("phi4 D=%d: %.2fx time, %.2fx allocations; value discrepancy %.3e\n",d,ot/nt,ob/nb,abs(a-b))
        push!(rows,join(("phi4",d,ot,nt,ot/nt,ob,nb,abs(a-b),norm(ga-gb)),','))
        oldrho = Legacy.get_ρ0_krylov(old;maxiter=200,maxdim=80)
        newrho = steady_state(s).rho
        ot,ob = measure(()->Legacy.get_ρ0_krylov(old;maxiter=200,maxdim=80))
        nt,nb = measure(()->steady_state(s))
        @printf("rho D=%d: %.2fx time; old residual %.2e, new residual %.2e\n",d,ot/nt,norm(lindblad(s,oldrho)),norm(lindblad(s,newrho)))
        push!(rows,join(("rho",d,ot,nt,ot/nt,ob,nb,norm(oldrho-newrho),0),','))
    end
    write(joinpath(@__DIR__,"performance.csv"),join(rows,'\n')*"\n")
end
main()
