@testset "Matrix-free kinetic resolvents" begin
    coherent=State(reshape(ComplexF64[.3+.1im],1,1),zeros(1,1))
    @test kinetic1(coherent;kinetic_backend=:krylov,gradient=false) ≈ .1 atol=1e-14
    @test kinetic2(coherent;kinetic_backend=:krylov,gradient=false) == 0
    @test iszero(norm(kinetic2(coherent;kinetic_backend=:krylov)[2]))

    # Force incremental basis growth and verify every physical shifted residual
    # independently of the Arnoldi identity used for its preliminary check.
    s=RandomState(8;seed=314,scaling=.3)
    before=(copy(s.R),copy(s.K),copy(s.Q),copy(s.ρ0))
    options=KineticKrylovOptions(initialdim=2,maxdim=64,tolerance=3e-12)
    ctx=RcmpsCore.kinetic_context(s,Val(:krylov),options)
    nodes=[0.,.01,.3,.8,1.]
    rhs=s.ρ0*s.R'
    for adj in (false,true)
        b=adj ? Matrix(s.R') : rhs
        x=RcmpsCore.kinetic_solutions(ctx,b,false,adj,nodes)
        for (i,c) in enumerate(nodes)
            matrix=reshape(x[:,i],s.D,s.D)
            action=adj ? RcmpsCore.lindblad_adjoint(s,matrix) : lindblad(s,matrix)
            @test norm(matrix-c*action-b)<=options.tolerance*norm(b)
        end
    end
    @test ctx.checked_solves==2length(nodes)
    @test ctx.maximum_relative_residual<=options.tolerance
    @test all(b.k<=options.maxdim && size(b.V,2)<=options.maxdim+1 for b in values(ctx.bases))

    terms=[(1.,RcmpsCore.kinetic_spec(false)),(1.,RcmpsCore.kinetic_spec(true))]
    dense=schur(generate_T(s))
    for n in (32,64)
        e,c,a=RcmpsCore.kinetic_sum(s,terms,ctx,n;gradient=true)
        ed,cd,ad=RcmpsCore.kinetic_sum(s,terms,dense,n;gradient=true)
        g=RcmpsCore.finish_gradient(s,c,a)
        gd=RcmpsCore.finish_gradient(s,cd,ad)
        @test e ≈ ed atol=1e-11 rtol=1e-11
        @test g ≈ gd atol=1e-8 rtol=1e-8
    end
    @test (s.R,s.K,s.Q,s.ρ0)==before

    small=RandomState(2;seed=702,scaling=.03)
    w=randn(MersenneTwister(103),ComplexF64,2,2);w/=norm(w)
    for objective in (kinetic1,kinetic2)
        e,g=objective(small;kinetic_backend=:krylov)
        ed,gd=objective(small;kinetic_backend=:resolvent)
        @test e ≈ ed atol=1e-12
        @test g ≈ gd atol=1e-8 rtol=1e-8
        h=1e-5
        f(t)=objective(first(retraction(small,w,t));kinetic_backend=:krylov,gradient=false)
        fd=(-f(2h)+8f(h)-8f(-h)+f(-2h))/(12h)
        @test abs(fd-metric(small,g,w))<1e-9
    end
    e,g=fg_phi4(small,2.;kinetic_backend=:krylov)
    ed,gd=fg_phi4(small,2.;kinetic_backend=:resolvent)
    @test e ≈ ed atol=1e-11
    @test g ≈ gd atol=1e-8 rtol=1e-8
    ep,gp=fg_phi4(small,2.;kinetic_backend=:krylov,parallel=:both)
    @test ep==e && gp==g
    small.model_param.name="phi4";small.model_param.coupling=2.
    optimized=optimize_state(small;n_iter=3,verbosity=0,diagnostics=true,
                             parallel=:both,kinetic_backend=:krylov)
    @test optimized.energy<e

    bad=KineticKrylovOptions(initialdim=1,maxdim=1,tolerance=1e-13)
    @test_throws SolverFailure kinetic1(s;kinetic_backend=:krylov,krylov_options=bad)
    @test_throws ArgumentError kinetic1(s;kinetic_backend=:krylov,
                                      krylov_options=KineticKrylovOptions(tolerance=Inf))
    @test_throws ArgumentError kinetic1(s;kinetic_backend=:krylov,
                                      krylov_options=KineticKrylovOptions(initialdim=0))
    # An unrealistically strict true residual is rejected, including at complete
    # basis exhaustion; it must never be silently accepted from the Ritz estimate.
    impossible=KineticKrylovOptions(initialdim=2,maxdim=4,tolerance=1e-30)
    @test_throws SolverFailure kinetic1(small;kinetic_backend=:krylov,krylov_options=impossible)
end

@testset "Checked Krylov fallback" begin
    s=RandomState(2;seed=702,scaling=.03)
    rhs=Matrix(s.R')
    # Both a capped one-vector basis and full basis exhaustion may use fallback.
    # A tiny shift makes the one-vector residual small but still nonzero.
    for maxdim in (1,4)
        options=KineticKrylovOptions(initialdim=1,maxdim=maxdim,
                                     tolerance=1e-30,fallback_tolerance=1e-10)
        ctx=RcmpsCore.kinetic_context(s,Val(:krylov),options)
        for adj in (false,true)
            nodes=[1e-12]
            x=RcmpsCore.kinetic_solutions(ctx,rhs,false,adj,nodes)
            matrix=reshape(x[:,1],s.D,s.D)
            action=adj ? RcmpsCore.lindblad_adjoint(s,matrix) : lindblad(s,matrix)
            residual=norm(matrix-nodes[1]*action-rhs)/norm(rhs)
            @test options.tolerance < residual <= options.fallback_tolerance
            basis=ctx.bases[(false,adj)]
            @test basis.exhausted || basis.k==maxdim
            @test maxdim==1 || basis.k>1 # do not accept early at the loose ceiling
        end
        @test ctx.fallback_solves==2
        @test ctx.checked_solves==2
    end
    strict=KineticKrylovOptions(initialdim=1,maxdim=1,tolerance=1e-30)
    ctx=RcmpsCore.kinetic_context(s,Val(:krylov),strict)
    @test_throws SolverFailure RcmpsCore.kinetic_solutions(ctx,rhs,false,false,[1e-12])
    # A genuinely underresolved solve must still fail with fallback enabled.
    loose=KineticKrylovOptions(initialdim=1,maxdim=1,fallback_tolerance=1e-10)
    ctx=RcmpsCore.kinetic_context(s,Val(:krylov),loose)
    @test_throws SolverFailure RcmpsCore.kinetic_solutions(ctx,rhs,false,false,[1.])
    @test ctx.checked_solves==0
    for ceiling in (Inf,NaN,0.,1e-13)
        @test_throws ArgumentError RcmpsCore.kinetic_context(s,Val(:krylov),
            KineticKrylovOptions(fallback_tolerance=ceiling))
    end
end

@testset "Backtracking preserves numerical failure cause" begin
    x=(R=ones(1,1),position=0.)
    retract=(x,p,alpha)->((R=x.R,position=x.position+alpha*p),p)
    inner=(x,g,p)->g*p
    fg=x->throw(SolverFailure("synthetic Krylov residual failure"))
    failure=try
        RcmpsCore.BackwardLineSearch(maxiter=3)(fg,x,-1.,(0.,1.);retract,inner)
        nothing
    catch e
        e
    end
    @test failure isa SolverFailure
    @test occursin("3 numerical trial failures",sprint(showerror,failure))
    @test occursin("synthetic Krylov residual failure",sprint(showerror,failure))
end
