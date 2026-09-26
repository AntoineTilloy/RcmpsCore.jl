# An independent finite-interval contraction: ordinary doubled ket/bra matrices,
# no connected transfer, no Poisson environments, no excitation source routines.
function finite_box_element(s,p,u,w,spec;decay_lengths=32)
    d=s.D; z=zeros(ComplexF64,d,d); id=Matrix{ComplexF64}(I,2d,2d)
    qk=[s.Q z; -s.R'*w s.Q-im*p*I]; rk=[s.R z; w s.R]
    qb=[s.Q z; -s.R'*u s.Q-im*p*I]; rb=[s.R z; u s.R]
    l=kron(id,qk)+kron(conj(qb),id)+kron(conj(rb),rk)
    ck=spec.commutator ? qk*rk-rk*qk : rk
    cb=spec.commutator ? qb*rb-rb*qb : rb
    sl,sr=kron(id,ck),kron(conj(cb),id)
    padding=decay_lengths/transfer_gap(s)
    prop=exp(padding*l)
    x=zeros(ComplexF64,2d,2d);x[1:d,1:d]=s.ρ0
    start=zeros(ComplexF64,4d^2,spec.channels);start[:,1]=prop*vec(x)
    function rhs!(dy,y,_,t)
        mul!(dy,l,y,2abs(t),0)
        for edge in spec.edges
            mul!(@view(dy[:,edge.dst]),edge.left ? sl : sr,@view(y[:,edge.src]),
                 edge.coefficient*RcmpsCore.transformed_source(t),1)
        end
    end
    a=sqrt(s.integrator_param.max_point)
    prob=RcmpsCore.ODEProblem(rhs!,start,(-a,a))
    sol=RcmpsCore.solve(prob,Vern7();tstops=[0.],abstol=1e-12,reltol=1e-12,
                        save_everystep=false,save_start=false)
    @test RcmpsCore.successful_retcode(sol)
    final=reshape(prop*sol.u[end][:,end],2d,2d)
    e=tr(final[1:d,1:d])
    return tr(final[d+1:2d,d+1:2d])-2*(padding+a^2)*tangent_metric(s,u,w)*e
end

@testset "Excitation free and coherent exact dispersion" begin
    vacuum=State(zeros(1,1),zeros(1,1))
    model=ExcitationModel(:phi4,0.)
    ps=[-2.,-.5,0.,.5,2.]
    disp=dispersion_relation(vacuum,ps;model)
    @test disp.energies[:,1] ≈ sqrt.(1 .+ ps.^2) atol=1e-11
    @test excitation_gap(vacuum;model) ≈ 1 atol=1e-12
    @test mass_gap(vacuum,model) ≈ 1 atol=1e-12
    @test transfer_gap(vacuum) == mass_gap(vacuum) == Inf
    diag=relativistic_diagnostics(disp)
    @test diag.c_squared ≈ 1 atol=1e-11
    @test diag.intercept ≈ 1 atol=1e-11
    @test diag.invariant_spread < 1e-10
    @test diag.fit_residual < 1e-12
    scaled=relativistic_diagnostics((momenta=ps,energies=2disp.energies);target_speed=2)
    @test scaled.invariant_mass_squared ≈ fill(4.,length(ps)) atol=1e-10
    @test scaled.c_squared ≈ 4 atol=1e-10
    for result in disp.spectra
        @test tangent_metric(vacuum,result.vectors[1],result.vectors[1]) ≈ 1
        @test maximum(result.residuals) < 1e-12
    end
    coherent=State(reshape([.3+.1im],1,1),zeros(1,1))
    coherent.integrator_param.max_point=30
    for p in (0.,.7,2.)
        omega=sqrt(1+p^2)
        e=excitation_gap(coherent;model=ExcitationModel(:phi4,2.))
        @test e ≈ 1+24*.3^2 atol=1e-8
        e=excitation_spectrum(coherent,p;model=ExcitationModel(:phi4,2.)).energies[1]
        @test e ≈ omega+24*.3^2/omega atol=1e-8
        for (name,beta) in ((:shg,.6),(:sg,.6im))
            e=excitation_spectrum(coherent,p;model=ExcitationModel(name,.6)).energies[1]
            exact=omega+real(cosh(beta*sqrt(2)*.3)-1)/(2omega)
            @test e ≈ exact atol=1e-8
        end
    end
    # Optimizer metric regularization must not affect physical excitation energies.
    vacuum.manifold_param.metric_reg=1e4
    @test excitation_gap(vacuum;model) ≈ 1 atol=1e-12
end

@testset "Tangent environments and independent finite-box contractions" begin
    s=RandomState(2;seed=82,scaling=.7)
    s.integrator_param.max_point=18
    s.integrator_param.atol=1e-12;s.integrator_param.rtol=1e-12
    u=randn(MersenneTwister(84),ComplexF64,2,2)
    w=randn(MersenneTwister(85),ComplexF64,2,2)
    for p in (0.,.6)
        ctx=RcmpsCore.ExcitationContext(s,p)
        pair=RcmpsCore.ExcitationPair(ctx,u,w)
        out=similar(pair.right)
        RcmpsCore.excitation_transfer!(out,pair.right,pair,similar(s.R))
        @test norm(out) < 1e-10
        @test abs(RcmpsCore.excitation_contract(pair,pair.right)) < 1e-11
        x=randn(MersenneTwister(86),ComplexF64,2,2,4)
        RcmpsCore.excitation_transfer!(out,x,pair,similar(s.R))
        @test abs(RcmpsCore.excitation_contract(pair,out)) < 1e-10
        env=excitation_environments(s,p,u,w)
        @test env.overlap ≈ tr(w*s.ρ0*u')
        @test norm(env.tail-boundary_environment(s,u'*w;method=:dense)) < 1e-10
        @test abs(tr(env.ket))+abs(tr(env.bra))+abs(tr(env.double)) < 1e-10
        for spec in (RcmpsCore.monomial_spec(2),RcmpsCore.monomial_spec(4),
                     RcmpsCore.kinetic_spec(true))
            exact=finite_box_element(s,p,u,w,spec)
            connected=RcmpsCore.excitation_observable(pair,spec)
            @test connected ≈ exact atol=3e-7 rtol=3e-8
        end
        # A constant density cancels exactly, even for off-diagonal complex tangents.
        @test abs(RcmpsCore.excitation_observable(pair,RcmpsCore.monomial_spec(0))) < 1e-10
    end
end

@testset "Excitation sesquilinearity, kinetic backends and eigenproblem" begin
    s=RandomState(2;seed=42,scaling=.5)
    s.integrator_param.max_point=30
    s.integrator_param.atol=1e-12;s.integrator_param.rtol=1e-12
    u=randn(MersenneTwister(1),ComplexF64,2,2)
    w=randn(MersenneTwister(2),ComplexF64,2,2)
    model=ExcitationModel(:phi4,1.)
    p=.7
    f(u,w;kw...)=excitation_matrix_element(s,p,u,w;model,kw...)
    a=f(u,w)
    @test a ≈ conj(f(w,u)) atol=1e-9
    @test f((2+im)*u,(.3-.7im)*w) ≈ (2-im)*(.3-.7im)*a atol=2e-8
    @test f(u,w+u) ≈ a+f(u,u) atol=2e-8
    @test a ≈ f(u,w;kinetic_backend=:ode) atol=2e-8
    @test f(u,u) ≈ f(u,u;kinetic_backend=:ode) atol=2e-8
    result=excitation_spectrum(s,p;model,nev=4)
    @test result.hermiticity_error < 1e-9
    @test maximum(result.residuals) < 1e-10
    @test issorted(result.energies)
    for i in 1:4, j in 1:4
        @test tangent_metric(s,result.vectors[i],result.vectors[j]) ≈ (i==j) atol=1e-10
    end
    @test f(result.vectors[1],result.vectors[1]) ≈ result.energies[1] atol=1e-8
    # A unitary auxiliary change of basis cannot change the excitation spectrum.
    gauge=Matrix(qr(randn(MersenneTwister(3),ComplexF64,2,2)).Q)
    transformed=State(gauge*s.R*gauge',gauge*s.K*gauge')
    copy_param!(transformed,s)
    @test excitation_spectrum(transformed,p;model,nev=4).energies ≈ result.energies atol=2e-8
    # Rank-deficient stationary density: discard null directions, don't shift rho.
    dark=State(ComplexF64[0 1;0 0],zeros(2,2))
    h=excitation_hamiltonian(dark,0.;model=ExcitationModel(:phi4,0.))
    @test h.metric_rank == 1
    @test size(h.hamiltonian) == (2,2)
end

@testset "Excitation input and numerical failures" begin
    s=State(zeros(1,1),zeros(1,1));model=ExcitationModel(:phi4,0.)
    @test_throws ArgumentError ExcitationModel(:unknown,1.)
    @test_throws ArgumentError ExcitationModel(:shg,0.)
    @test_throws ArgumentError ExcitationModel(:phi4,NaN)
    s.model_param.name="phi4_m"
    @test_throws ArgumentError ExcitationModel(s)
    @test ExcitationModel(s;massmod=.8).name == :phi4_m
    s.model_param.name="phi4"
    @test_throws ArgumentError excitation_spectrum(s,NaN;model)
    @test_throws ArgumentError excitation_spectrum(s,0.;model,nev=0)
    @test_throws ArgumentError excitation_spectrum(s,0.;model,nev=2)
    @test_throws ArgumentError excitation_hamiltonian(s,0.;model,metric_tol=-1.)
    @test_throws ArgumentError excitation_hamiltonian(s,0.;model,max_bond=0)
    @test_throws ArgumentError excitation_gap(s;model,kinetic_backend=:invalid)
    @test_throws DimensionMismatch excitation_matrix_element(s,0.,zeros(2,2),zeros(1,1);model)
    @test_throws ArgumentError dispersion_relation(s,Float64[];model)
    @test_throws ArgumentError relativistic_diagnostics((momenta=[-1.,1.],energies=ones(2,1)))
    @test_throws ArgumentError relativistic_diagnostics((momenta=[0.,1.],energies=-ones(2,1)))
    @test_throws ArgumentError relativistic_diagnostics((momenta=[0.,1.],energies=ones(2,1));target_speed=0)
    @test_throws SolverFailure excitation_spectrum(s,2.;model,
        kinetic_options=KineticOptions(minnodes=2,maxnodes=4,atol=1e-15,rtol=0.))
    s.integrator_param.maxiters=1
    @test_throws SolverFailure excitation_gap(s;model,kinetic_backend=:ode)
end
