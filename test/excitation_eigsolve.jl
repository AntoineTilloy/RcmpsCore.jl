const RC = RcmpsCore

@testset "Restarted Hermitian Davidson" begin
    rng=MersenneTwister(7);n=40
    U=Matrix(qr(randn(rng,ComplexF64,n,n)).Q)
    values=vcat([.3,.5,.9],collect(range(2,30,length=n-3)))
    A=Hermitian(U*Diagonal(values)*U')
    x0=randn(MersenneTwister(1),ComplexF64,n)
    # Diagonal preconditioner from a perturbed operator, as for the coarse kinetic matrix.
    P=Hermitian(U*Diagonal(values.*(1 .+ .2*rand(MersenneTwister(2),n)))*U')
    for (refresh,maxspace) in ((true,8),(false,16)), nev in (1,3), precondition in (identity,r->P\r)
        o=DavidsonOptions(;tol=1e-9,maxops=400,maxspace,retain=refresh ? nev : nev+3,refresh,stall_window=0)
        r=RC.davidson(x->A*x,x0;nev,precondition,options=o)
        @test r.converged && r.stop_reason==:converged
        @test r.energies ≈ values[1:nev] atol=1e-10
        for k in 1:nev
            @test norm(A*r.vectors[k]-r.energies[k]*r.vectors[k]) ≈ r.residuals[k] atol=1e-12
            @test r.actions[k] ≈ A*r.vectors[k] atol=1e-12
        end
        @test length(r.history)==r.numops
    end
    # Fresh restarts recompute the retained action and record how much it changed.
    calls=Ref(0)
    r=RC.davidson(x->(calls[]+=1;A*x),x0;options=DavidsonOptions(;tol=1e-9,maxops=400,stall_window=0))
    @test calls[]==r.numops
    @test any(row->row.cycle>1,r.history)
    @test maximum(row.refresh_change for row in r.history) < 1e-12
    limited=RC.davidson(x->A*x,x0;options=DavidsonOptions(;maxops=2,stall_window=0))
    @test limited.stop_reason==:maxops && !limited.converged
    permissive=DavidsonOptions(;tol=1e-14,maxops=60,maxspace=16,retain=4,refresh=false,stall_window=2,
        stall_minops=3,stall_rtol=.99,stall_residual_limit=100.,stall_energy_atol=100.)
    early=RC.davidson(x->A*x,x0;options=permissive)
    @test early.stop_reason==:stalled && !early.converged
    @test_throws SolverFailure RC.davidson(x->(1+1e-4im)*x,ones(3))
    @test_throws ArgumentError RC.davidson(x->x,ones(3);options=DavidsonOptions(;maxspace=1))
    @test_throws ArgumentError RC.davidson(x->x,ones(3);nev=4,options=DavidsonOptions(;maxspace=8))
    @test_throws ArgumentError RC.davidson(x->x,zeros(3))
    # At most maxspace-1 start vectors; loose targets need no explicit stall limit.
    @test_throws ArgumentError RC.davidson(x->A*x,[randn(MersenneTwister(k),ComplexF64,n) for k in 1:4];
        options=DavidsonOptions(;maxspace=4))
    @test RC.check_davidson(DavidsonOptions(;tol=1e-4),1)==1
    @test RC.check_davidson(DavidsonOptions(;tol=1e-4,stall_window=0),1)==1
    @test RC.stall_limit(DavidsonOptions(;tol=1e-4))==1e-3
    # An inexact (slightly non-Hermitian) operator whose whole space is spanned returns
    # its Ritz pairs as :breakdown instead of throwing.
    H=Hermitian(randn(MersenneTwister(8),ComplexF64,6,6));S=randn(MersenneTwister(9),ComplexF64,6,6);S=S-S'
    broken=RC.davidson(x->(H+1.5e-7*S)*x,ones(6);options=DavidsonOptions(;tol=1e-12,hermitian_tol=1e-5,stall_window=0))
    @test broken.stop_reason==:breakdown && !broken.converged
    @test broken.energies[1] ≈ eigmin(H) atol=1e-5
    # Stall rule: synthetic histories (ported from the research solver tests).
    o=DavidsonOptions(;stall_residual_limit=1e-6,stall_energy_atol=1e-10)
    stallrows(es,rs)=[(energies=[e],residual=r) for (e,r) in zip(es,rs)]
    @test RC.davidson_stalled(stallrows(fill(.02,40),fill(3e-7,40)),o)
    @test !RC.davidson_stalled(stallrows(fill(.02,39),fill(3e-7,39)),o)
    @test !RC.davidson_stalled(stallrows(fill(.02,80),fill(1e-4,80)),o)
    @test !RC.davidson_stalled(stallrows(.02 .+ 1e-8 .* (1:40),fill(3e-7,40)),o)
    @test !RC.davidson_stalled(stallrows(fill(.02,80),1e-6 .* .97 .^ (1:80)),o)
    @test !RC.davidson_stalled(stallrows(fill(.02,40),fill(3e-7,40)),DavidsonOptions(;stall_window=0))
    # Two roots after fresh restarts: rows rebuilding the space carry only the converged
    # first root and must not make a still-improving second root look stalled.
    mixed=NamedTuple[]
    for k in 1:60
        push!(mixed,(energies=[.1],residual=1e-9))
        push!(mixed,(energies=[.1,.3],residual=5e-7*.9^k))
    end
    @test !any(k->RC.davidson_stalled(mixed[1:2k],o),1:60)
    flat2=[(energies=[.1,.3],residual=3e-7) for _ in 1:40]
    drift2=[(energies=[.1,.3+1e-8k],residual=3e-7) for k in 1:40]
    @test RC.davidson_stalled(flat2,o) && !RC.davidson_stalled(drift2,o)
    # A :maxops stop never truncates the targeted roots, even during a fresh rebuild.
    for maxops in 2:13
        r=RC.davidson(x->A*x,x0;nev=2,options=DavidsonOptions(;tol=1e-14,maxspace=4,maxops,stall_window=0))
        @test length(r.energies)==2 && r.numops<=maxops
    end
    @test_throws ArgumentError RC.davidson(x->A*x,x0;nev=3,options=DavidsonOptions(;maxops=2))
    # Small start vectors are scaled, not rejected; the callback sees the documented shapes.
    shapes=Tuple[]
    r=RC.davidson(x->A*x,1e-9*x0;nev=2,options=DavidsonOptions(;tol=1e-9,maxops=400,stall_window=0),
        callback=(row,u,au)->push!(shapes,(row.dimension,size(u),size(au))))
    @test r.converged
    @test r.energies ≈ values[1:2] atol=1e-10
    @test all(q->q[1]<=8 && q[2]==q[3] && q[2][1]==n && q[2][2]==min(2,q[1]),shapes)
    # Breakdown is declared only when no unconverged root can expand: a floor-limited root 1
    # (anti-Hermitian part in its 3-dim block) must not stop root 2 from converging.
    U1=Matrix(qr(randn(MersenneTwister(11),ComplexF64,3,3)).Q);S1=randn(MersenneTwister(12),ComplexF64,3,3)
    U2=Matrix(qr(randn(MersenneTwister(13),ComplexF64,60,60)).Q)
    B1=U1*Diagonal([.1,30.,40.])*U1'+3e-6*(S1-S1');B2=U2*Diagonal(vcat(.5,range(1,20,length=59)))*U2'
    blockop(x)=vcat(B1*x[1:3],B2*x[4:end])
    # The floor-limited block is spanned from the start, so root 1 is present but cannot expand.
    starts=vcat([vcat((1:3) .== j,zeros(60)) for j in 1:3],[vcat(zeros(3),randn(MersenneTwister(15),ComplexF64,60))])
    r=RC.davidson(blockop,starts;nev=2,options=DavidsonOptions(;tol=1e-7,maxspace=80,retain=4,refresh=false,
        hermitian_tol=1e-5,stall_window=0,maxops=300))
    @test r.energies[1] ≈ .1 atol=1e-5
    @test r.energies[2] ≈ .5 atol=1e-9
    @test r.residuals[1] > 1e-7 && r.residuals[2] <= 1e-7 && r.stop_reason==:breakdown
    # The stall rule on a captured D=48 residual history stops at its plateau only.
    path=joinpath(@__DIR__,"fixtures","g2725_D48_first85.csv")
    rows=[split(line,',') for line in readlines(path)[2:end]]
    history=[(energies=[parse(Float64,q[2])],residual=parse(Float64,q[3])) for q in rows]
    o=DavidsonOptions(;stall_residual_limit=1e-6,stall_energy_atol=1e-10)
    first=findfirst(k->RC.davidson_stalled(history[1:k],o),eachindex(history))
    @test !isnothing(first) && 50<=first<=65
end

function z2_test_state()
    s=RandomStateZ2(4;seed=11,scaling=.5)
    s.integrator_param.max_point=14;s.integrator_param.n_quad=48
    s.integrator_param.atol=1e-11;s.integrator_param.rtol=1e-10
    s.model_param.name="phi4";s.model_param.coupling=1.
    return s
end

@testset "Parity sectors and white coordinates" begin
    s=z2_test_state()
    z=excitation_parity(s)
    @test z==[1,1,-1,-1]
    @test isnothing(excitation_parity(RandomState(4;seed=3)))
    @test_throws ArgumentError excitation_parity(s,[1,-1,1,-1])
    @test_throws ArgumentError excitation_parity(s,[1,1,1,1])
    Z=Diagonal(z)
    for gauge in (:density,:original), (sector,dimension) in ((:odd,8),(:even,8),(:all,16))
        c=TangentCoordinates(s;sector,gauge)
        @test length(c)==dimension && c.sector==sector
        @test c.metric_defect < 1e-12
        x,y=randn(MersenneTwister(2),ComplexF64,dimension),randn(MersenneTwister(3),ComplexF64,dimension)
        wx,wy=RC.tangent_matrix(c,x),RC.tangent_matrix(c,y)
        @test tangent_metric(c.state,wy,wx) ≈ dot(y,x) atol=1e-12
        @test RC.tangent_coordinates(c,wx) ≈ x atol=1e-12
        wo=RC.original_tangent(c,wx)
        @test tangent_metric(s,RC.original_tangent(c,wy),wo) ≈ dot(y,x) atol=1e-12
        sector==:odd && @test norm(Z*wo*Z-wo) < 1e-14
        sector==:even && @test norm(Z*wo*Z+wo) < 1e-14
        @test RC.coordinate_tangent(c,wo) ≈ wx atol=1e-12
    end
    @test TangentCoordinates(s).sector==:odd
    @test TangentCoordinates(RandomState(3;seed=4)).sector==:all
    @test_throws ArgumentError TangentCoordinates(RandomState(3;seed=4);sector=:odd)
    @test_throws ArgumentError TangentCoordinates(s;gauge=:invalid)
    @test_throws ArgumentError TangentCoordinates(s;metric_tol=1.)
    # Rank-deficient density: unresolved directions are dropped, never regularized.
    dark=State(ComplexF64[0 1;0 0],zeros(2,2))
    for gauge in (:density,:original)
        c=TangentCoordinates(dark;sector=:all,gauge,metric_tol=1e-12)
        @test c.metric_rank==1 && length(c)==2
    end
    # Rotated dark states have a roundoff eigenvalue of either sign. The Float64 floor
    # drops it even at the default metric_tol=0, in both gauges.
    for seed in 1:6
        g=Matrix(qr(randn(MersenneTwister(seed),2,2)).Q)
        rotated=State(g'*ComplexF64[0 1;0 0]*g,zeros(2,2))
        for gauge in (:density,:original)
            c=TangentCoordinates(rotated;sector=:all,gauge)
            @test c.metric_rank==1 && length(c)==2
            x=randn(MersenneTwister(seed),ComplexF64,2)
            w=RC.tangent_matrix(c,x)
            @test real(tangent_metric(c.state,w,w)) ≈ norm(x)^2 atol=1e-12
            @test RC.tangent_coordinates(c,w) ≈ x atol=1e-12
        end
    end
    free=ExcitationModel(:phi4,0.)
    g=Matrix(qr(randn(MersenneTwister(3),2,2)).Q)
    rotated=State(g'*ComplexF64[0 1;0 0]*g,zeros(2,2))
    reference=eigvals(excitation_hamiltonian(rotated,0.;model=free,assembly=:action,kinetic_nodes=64).hamiltonian)
    r=excitation_eigsolve(rotated,0.;model=free,nev=1,kinetic_nodes=64,parallel=:serial)
    @test r.dimension==2
    @test r.energies[1] ≈ reference[1] atol=1e-9
end

@testset "Sector Hamiltonians and the iterative solver" begin
    s=z2_test_state()
    model=ExcitationModel(:phi4,1.)
    nodes=32
    for p in (0.,.6)
        h(sector)=eigvals(excitation_hamiltonian(s,p;model,sector,assembly=:action,kinetic_nodes=nodes).hamiltonian)
        full,odd,even=h(:all),h(:odd),h(:even)
        @test sort(vcat(odd,even)) ≈ full atol=1e-9
        for (sector,reference) in ((:odd,odd),(:even,even)), gauge in (:density,:original)
            r=excitation_eigsolve(s,p;model,sector,gauge,nev=2,kinetic_nodes=nodes,parallel=:serial,
                davidson=DavidsonOptions(;tol=1e-8,maxops=60,maxspace=10))
            @test r.converged
            @test r.energies ≈ reference[1:2] atol=1e-8
            @test r.start==(sector==:odd ? :field : :kinetic)
            @test r.preconditioner.skew < 1e-8
            for (k,v) in enumerate(r.validation)
                @test v.standard_energy ≈ r.energies[k] atol=1e-8
                @test abs(v.bilinear_energy-v.standard_energy) < 1e-8
                @test abs(v.refined_bilinear_energy-v.refined_energy) < 1e-8
                @test v.max_imaginary < 1e-8
            end
            for k in 1:2, l in 1:2
                @test tangent_metric(s,r.vectors[k],r.vectors[l]) ≈ (k==l) atol=1e-10
            end
            @test excitation_matrix_element(s,p,r.vectors[1],r.vectors[1];model,kinetic_nodes=nodes) ≈
                r.energies[1] atol=1e-8
        end
    end
    # Momentum reuse shares the Schur and Poisson factors and reproduces a fresh context.
    base=ExcitationContext(s,0.;max_bond=4,store_transfer=false)
    moved=ExcitationContext(base;momentum=.6)
    w=randn(MersenneTwister(5),ComplexF64,4,4)
    @test excitation_hamiltonian_action(moved,w;model,kinetic_nodes=nodes) ≈
        excitation_hamiltonian_action(ExcitationContext(s,.6),w;model,kinetic_nodes=nodes) atol=1e-10
    @test moved.poisson===base.poisson && moved.schur===base.schur
    tight=ExcitationContext(base;integrator=IntegratorParameters(max_point=20))
    @test tight.state.integrator_param.max_point==20 && base.state.integrator_param.max_point==14
    @test base.ketfactor===base.poisson
    # Fixed kinetic nodes agree with adaptive refinement once resolved.
    @test excitation_hamiltonian_action(base,w;model,kinetic_nodes=256) ≈
        excitation_hamiltonian_action(base,w;model) atol=1e-8
    @test_throws ArgumentError excitation_hamiltonian_action(base,w;model,kinetic_nodes=1)
    # Dispersion reuses factors and warm-starts; it agrees with the dense sector reference.
    disp=dispersion_relation(s,[0.,.6];model,solver=:iterative,kinetic_nodes=nodes,parallel=:serial,
        davidson=DavidsonOptions(;tol=1e-8,maxops=60,maxspace=10))
    dense=dispersion_relation(s,[0.,.6];model,sector=:odd,assembly=:action,kinetic_nodes=nodes)
    # Reported dispersion energies are the refined validated ones; compare the solver
    # levels with the dense matrix at the same (standard) settings.
    @test [r.energies[1] for r in disp.spectra] ≈ dense.energies[:,1] atol=1e-8
    @test disp.energies[:,1] ≈ [r.validated_energies[1] for r in disp.spectra]
    @test last(disp.spectra).start==:supplied
    @test excitation_gap(s;model,solver=:iterative,kinetic_nodes=nodes,parallel=:serial) ≈
        disp.energies[1] atol=1e-8
    # Exact free dispersion in the one-dimensional vacuum tangent space.
    vacuum=State(zeros(1,1),zeros(1,1))
    for p in (0.,.7)
        r=excitation_eigsolve(vacuum,p;model=ExcitationModel(:phi4,0.))
        @test r.sector==:all && r.dimension==1 && r.converged
        @test only(r.energies) ≈ sqrt(1+p^2) atol=1e-11
    end
    # Field seed: the momentum-shifted gauge transformation removes the left-gauge defect of
    # the raw insertion (R,1) at every p, and the seed lies in the odd sector.
    for p in (0.,.6)
        c=TangentCoordinates(s);ctx=ExcitationContext(c.state,p);t=c.state
        x=-RC.action_dual_poisson(ctx,t.R+t.R';shifted=true)
        w=Matrix{ComplexF64}(I,4,4)+t.R*x-x*t.R
        @test norm(t.R+t.Q*x-x*t.Q-im*p*x+t.R'*w) < 1e-9
        op=RC.ExcitationOperator(ctx,c,model;kinetic_nodes=nodes)
        @test RC.tangent_matrix(c,RC.field_seed(op)) ≈ w atol=1e-10
        # The coarse preconditioner is the kinetic-only sector matrix on its grid.
        pre=RC.kinetic_preconditioner(op;nodes=8,threaded=false)
        kinetic=excitation_hamiltonian(s,p;model=ExcitationModel(:phi4,0.),sector=:odd,gauge=:density,
            metric_tol=0.,assembly=:action,kinetic_nodes=8).hamiltonian
        @test Hermitian((pre.matrix+pre.matrix')/2) ≈ kinetic atol=1e-10
        @test pre.floor==1.
    end
    # Validation settings, acceptance, nev>1 with default options, refine_steps.
    r=excitation_eigsolve(s,0.;model,nev=4,kinetic_nodes=nodes,parallel=:serial)
    @test r.converged && r.accepted && r.davidson.maxspace==10
    @test r.refined_integration.max_point==24 && r.refined_integration.n_quad==96
    @test r.refined_integration.atol ≈ 1e-12 && r.refined_kinetic_nodes==2nodes
    @test all(v->v.accepted && v.consistent,r.validation)
    loose=excitation_eigsolve(s,0.;model,kinetic_nodes=nodes,parallel=:serial,refine_steps=3,
        davidson=DavidsonOptions(;tol=1e-2,maxops=3,stall_window=0),start=:random)
    @test !isempty(loose.corrections) && first(loose.corrections).step==0
    @test loose.energies[1] ≈ loose.validation[1].refined_energy atol=1e-10
    @test loose.residuals[1] ≈ loose.validation[1].refined_residual atol=1e-10
    @test loose.residuals[1] <= first(loose.corrections).residual
    @test loose.converged==(loose.stop_reason==:converged) && loose.refined_converged isa Bool
    @test_throws SolverFailure excitation_gap(s;model,solver=:iterative,kinetic_nodes=nodes,
        parallel=:serial,validation_tol=1e-16)
    # A larger sector forces restarts: fresh and thick restarts agree with the dense matrix.
    s6=RandomStateZ2(6;seed=21,scaling=.5)
    s6.integrator_param.max_point=14;s6.integrator_param.n_quad=48
    s6.integrator_param.atol=1e-11;s6.integrator_param.rtol=1e-10
    odd6=eigvals(excitation_hamiltonian(s6,0.;model,sector=:odd,assembly=:action,kinetic_nodes=nodes).hamiltonian)
    for options in (DavidsonOptions(;tol=1e-8,maxspace=6,stall_window=0),
                    DavidsonOptions(;tol=1e-8,maxspace=8,retain=3,refresh=false,stall_window=0))
        r=excitation_eigsolve(s6,0.;model,nev=2,kinetic_nodes=nodes,parallel=:serial,davidson=options,validate=false)
        @test r.converged && r.dimension==18
        @test any(row->row.cycle>1,r.history)
        @test r.energies ≈ odd6[1:2] atol=1e-8
    end
    # A state without parity at nonzero momentum, and a supplied tangent start.
    s3=RandomState(3;seed=5,scaling=.5)
    s3.integrator_param.max_point=14;s3.integrator_param.n_quad=48
    s3.integrator_param.atol=1e-11;s3.integrator_param.rtol=1e-10
    dense3=excitation_spectrum(s3,.5;model,nev=2,assembly=:action,kinetic_nodes=nodes)
    r=excitation_eigsolve(s3,.5;model,nev=2,kinetic_nodes=nodes,parallel=:serial,validate=false)
    @test r.sector==:all && r.dimension==9 && r.converged
    @test r.energies ≈ dense3.energies atol=1e-8
    warm=excitation_eigsolve(s3,.5;model,kinetic_nodes=nodes,parallel=:serial,validate=false,
        start=dense3.vectors[1])
    @test warm.start==:supplied && warm.numops <= 2
    @test warm.energies[1] ≈ dense3.energies[1] atol=1e-8
    # strict=false keeps every momentum and reports acceptance; strict fails fast.
    lax=dispersion_relation(s,[0.,.6];model,solver=:iterative,kinetic_nodes=nodes,parallel=:serial,
        validation_tol=1e-16,strict=false)
    @test lax.accepted==[false,false] && size(lax.energies)==(2,1)
    @test_throws SolverFailure dispersion_relation(s,[0.,.6];model,solver=:iterative,kinetic_nodes=nodes,
        parallel=:serial,validation_tol=1e-16)
    # Cheap requests fail before any dense factorization.
    @test_throws ArgumentError excitation_eigsolve(s;model,start=[randn(ComplexF64,8) for _ in 1:8])
    @test_throws DimensionMismatch excitation_eigsolve(s;model,start=randn(ComplexF64,7))
    @test_throws ArgumentError excitation_eigsolve(s;model,refined_kinetic_nodes=1)
    @test_throws ArgumentError excitation_eigsolve(s;model,validate=1)
    @test_throws ArgumentError excitation_eigsolve(s;model,nev=2,davidson=DavidsonOptions(;maxspace=2))
    @test_throws ArgumentError excitation_eigsolve(s;model,kinetic_node=3)
    @test_throws ArgumentError excitation_eigsolve(s;model,max_bond=2)
    @test_throws ArgumentError excitation_eigsolve(s;model,nev=9)
    @test_throws ArgumentError excitation_eigsolve(s;model,refine_steps=1,nev=2)
    @test_throws ArgumentError excitation_eigsolve(s;model,sector=:even,start=:field)
    @test_throws ArgumentError excitation_gap(s;model,solver=:invalid)
end
