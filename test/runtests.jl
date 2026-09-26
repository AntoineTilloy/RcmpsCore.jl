using RcmpsCore, Test, LinearAlgebra, Random
using OrdinaryDiffEqVerner: Vern7
BLAS.set_num_threads(1)

@testset "Wolfe acceptance at the objective noise floor" begin
    x=(R=ones(1,1),position=0.0)
    retract=(x,p,alpha)->((R=x.R,position=x.position+alpha*p),p)
    inner=(x,g,p)->g*p
    # Accurate derivative, but a value perturbation larger than the true decrease.
    fg=x->((x.position-1e-6)^2/2+(iszero(x.position) ? 0. : 1e-11),x.position-1e-6)
    ls=SafeguardedLineSearch(energy_atol=1e-10)
    y,f,g,_,alpha,_=ls(fg,x,1e-6;retract,inner,acceptfirst=true)
    @test y.position ≈ 1e-6
    @test g == 0
    @test fg(x)[1] < f <= fg(x)[1]+ls.energy_atol
    @test alpha>0
    @test_throws SolverFailure SafeguardedLineSearch(energy_atol=0)(fg,x,1e-6;retract,inner,acceptfirst=true)
    @test_throws SolverFailure ls(fg,x,-1e-6;retract,inner,acceptfirst=true)
    @test_throws ArgumentError SafeguardedLineSearch(energy_atol=-1.)
end

@testset "Canonical state and transfer" begin
    s = RandomState(3;seed=61)
    @test s.Q+s.Q'+s.R'*s.R ≈ zeros(3,3) atol=1e-13
    @test RandomState(3;seed=61).R == s.R
    @test_throws ArgumentError RandomStateZ2(3)
    @test_throws ArgumentError State(zeros(2,3),zeros(2,2))
    @test_throws ArgumentError State(ones(2,2),[0 1;0 0])
    @test_throws SolverFailure State(zeros(2,2),zeros(2,2))
    x = randn(MersenneTwister(1),ComplexF64,3,3)
    y = randn(MersenneTwister(2),ComplexF64,3,3)
    @test vec(lindblad(s,x)) ≈ generate_T(s)*vec(x)
    @test dot(y,lindblad(s,x)) ≈ dot(lindblad_adjoint(s,y),x)
    out,tmp = similar(x),similar(x)
    @test lindblad!(out,s,x,tmp) ≈ lindblad(s,x)
    @test lindblad_adjoint!(out,s,x,tmp) ≈ lindblad_adjoint(s,x)
    @test abs(tr(lindblad(s,x))) < 1e-12
    @test mass_gap(s;method=:iterative) ≈ mass_gap(s;method=:dense) rtol=1e-8
    r,k = copy(s.R),copy(s.K)
    t = State(r,k)
    r[1,1] += 1
    @test t.R == s.R
    w = randn(MersenneTwister(3),ComplexF64,3,3)
    epsstep = 1e-6
    a,_ = retraction(s,w,epsstep)
    b,_ = retraction(s,w,-epsstep)
    @test (a.Q-b.Q)/(2epsstep) ≈ -s.R'*w rtol=1e-9
    a.integrator_param.max_point = 4
    @test s.integrator_param.max_point == 20
end

@testset "CPTP stationary solver and boundary Poisson equation" begin
    for d in (2,5,12)
        s = RandomState(d;seed=d)
        exact = steady_state(s;method=:dense)
        iter = steady_state(s;method=:cptp)
        @test norm(iter.rho-exact.rho) < 1e-9
        @test norm(lindblad(s,iter.rho)) < 1e-8
        @test tr(iter.rho) ≈ 1
        @test eigmin(Hermitian(iter.rho)) >= -1e-12
        a = randn(MersenneTwister(12),ComplexF64,d,d)
        a += a'
        h = boundary_environment(s,a)
        hd = boundary_environment(s,a;method=:dense)
        @test h ≈ hd atol=1e-8 rtol=1e-8
        if d == 5
            # Force a short Krylov attempt and exercise the checked small-D fallback.
            hf = boundary_environment(s,a;krylovdim=2,maxiter=1)
            @test hf ≈ hd atol=1e-8 rtol=1e-8
        end
        @test abs(tr(s.ρ0*h)) < 1e-9
        @test -lindblad_adjoint(s,h) ≈ a-tr(s.ρ0*a)*I atol=1e-8
    end
    # A defective no-jump matrix must be handled by Schur, not eigenvector inversion.
    r = ComplexF64[0 1;0 0]
    dark = State(r,zeros(2,2))
    @test dark.ρ0 ≈ [1 0;0 0] atol=1e-10
    @test entanglement_entropy(dark) < 1e-8
    @test entanglement_entropy(dark;alpha=2) < 1e-8
    @test_throws SolverFailure State(r,zeros(2,2);ρ0=[-1 0;0 2])
    q = ComplexF64[-1 1;0 -1]
    p = RcmpsCore.NoJumpInverse(q,Matrix{ComplexF64}(I,2,2))
    y = ComplexF64[2 im;-im 1]
    x = p(y)
    @test -(q*x+x*q') ≈ y
    x = p(y;adjoint=true)
    @test -(q'*x+x*q) ≈ y
end

@testset "Source, quadrature and coherent-state exact values" begin
    @test RcmpsCore.transformed_source(0.) ≈ 1/sqrt(pi)
    @test RcmpsCore.transformed_source(1e-10) ≈ 1/sqrt(pi) rtol=1e-8
    s = State(reshape([.3+.1im],1,1),zeros(1,1))
    s.integrator_param.max_point = 30
    s.integrator_param.atol = 1e-12
    s.integrator_param.rtol = 1e-12
    for n in 0:4
        @test phi_monomial(s,n;gradient=false) ≈ (sqrt(2)*.3)^n atol=1e-10
    end
    @test vertex(s,.7;gradient=false) ≈ exp(.7*sqrt(2)*.3) atol=1e-10
    @test vertex(s,.7im;gradient=false) ≈ exp(.7im*sqrt(2)*.3) atol=1e-10
    @test kinetic1(s;gradient=false) ≈ .1 atol=1e-10
    @test kinetic2(s;gradient=false) ≈ 0 atol=1e-12
end

function directional_check(s,f,w;rtol=2e-5,atol=1e-7)
    value,g = f(s)
    h = 1e-5
    plus,_ = retraction(s,w,h)
    minus,_ = retraction(s,w,-h)
    fd = real(f(plus;gradient=false)-f(minus;gradient=false))/(2h)
    @test metric(s,g,w) ≈ fd rtol=rtol atol=atol
    @test real(value) ≈ real(f(s;gradient=false)) atol=1e-10
    return abs(metric(s,g,w)-fd)
end
@testset "Independent finite differences of every gradient" begin
    base = RandomState(3;seed=182)
    s = State(.12base.R,.12^2*base.K)
    s.integrator_param.atol = 2e-12
    s.integrator_param.rtol = 2e-12
    s.integrator_param.max_point = 12
    w = randn(MersenneTwister(7),ComplexF64,3,3)
    w /= norm(w)
    for f in ((s;kw...)->vertex(s,.4;kw...), (s;kw...)->vertex(s,.4im;kw...),
              (s;kw...)->phi_monomial(s,1;kw...), (s;kw...)->phi_monomial(s,2;kw...),
              (s;kw...)->phi_monomial(s,4;kw...), kinetic1, kinetic2,
              (s;kw...)->fg_phi4(s,1.2;kw...), (s;kw...)->fg_sgordon(s,.5;kw...),
              (s;kw...)->fg_sgordon(s,.5im;kw...),
              (s;kw...)->fg_phi4_massmod(s,1.2,.8;kw...))
        directional_check(s,f,w)
    end
    f(s;kw...) = vertex(s,.4;kw...)
    h = 1e-5
    plus,_ = retraction(s,w,h); minus,_ = retraction(s,w,-h)
    fd = real(f(plus;gradient=false)-f(minus;gradient=false))/(2h)
    _,full = f(s)
    _,bulk = f(s;boundary=false)
    @test abs(metric(s,bulk,w)-fd) > 1e-3
    @test abs(metric(s,full,w)-fd) < 1e-7
    @test full ≈ f(s;boundary_method=:dense)[2] atol=1e-9
    # In-place RHS is allocation-free after compilation, with contiguous channel views.
    spec = RcmpsCore.kinetic_spec(false)
    ws = RcmpsCore.ODEWorkspace{false}(s,spec,s.R,similar(s.R))
    u = randn(ComplexF64,3,3,4); du = similar(u)
    RcmpsCore.ode_rhs!(du,u,ws,.4)
    @test @allocated(RcmpsCore.ode_rhs!(du,u,ws,.4)) == 0
    s.integrator_param.maxiters = 1
    @test_throws SolverFailure vertex(s,.4)
end

@testset "Storage and optimization" begin
    s = RandomState(2;seed=4,scaling=.2)
    s.model_param.name = "phi4"
    s.model_param.coupling = 1.0
    s.info = "round trip"
    mktempdir() do dir
        path = rcmps_store(s,joinpath(dir,"state.jld2");verbose=false)
        t = rcmps_load(path)
        @test t.R == s.R && t.K ≈ s.K && t.info == s.info
        @test t.ρ0 ≈ s.ρ0
        @test length(get_states(dir)) == 1
        @test get_exact_state([s],"missing",0,2) === nothing
        s.e0 = 2
        @test get_exact_state([s],"phi4",1,2) === s
    end
    initial = fg_phi4(s,1.;gradient=false)
    for method in ("GD","CG","LBFGS")
        result = optimize_state(s;method,n_iter=2,verbosity=0,ls=ls_backward,diagnostics=true)
        @test result.energy < initial
        @test isfinite(result.gradient_norm)
    end
    @test_throws ArgumentError optimize_state(s;method="bogus")
end

@testset "Convergence controls and concurrent evaluations" begin
    s = RandomState(3;seed=16,scaling=.3)
    s.integrator_param.atol = 1e-12
    s.integrator_param.rtol = 1e-12
    s.integrator_param.n_quad = 64
    v64,g64 = fg_phi4(s,1.)
    s.integrator_param.n_quad = 128
    v128,g128 = fg_phi4(s,1.)
    @test v64 ≈ v128 atol=1e-10
    @test g64 ≈ g128 atol=2e-8 rtol=1e-7
    s.integrator_param.max_point = 30
    v30,g30 = fg_phi4(s,1.)
    @test v30 ≈ v128 atol=1e-8
    @test g30 ≈ g128 atol=1e-7 rtol=1e-6
    a = Threads.@spawn phi_monomial(s,2)
    b = Threads.@spawn phi_monomial(s,2)
    va,ga = fetch(a); vb,gb = fetch(b)
    @test va ≈ vb atol=1e-12
    @test ga ≈ gb atol=1e-12
    s.model_param.name = "phi4"
    # Exercise the released default Wolfe line search as well as custom Armijo.
    result = optimize_state(s;n_iter=2,verbosity=0,diagnostics=true)
    @test result.energy < v30
end

@testset "Line search rejects failed trials" begin
    s = State(reshape(ComplexF64[0.3],1,1),zeros(1,1))
    attempts = Ref(0)
    objective(x) = begin
        attempts[] += 1
        abs(x.R[1]) > 0.5 && throw(SolverFailure("synthetic invalid trial"))
        return abs2(x.R[1]),x.R/(1+x.manifold_param.metric_reg)
    end
    f,g = objective(s)
    direction = -4g
    y,fy,gy,v,alpha,nfg = ls_backward(objective,s,direction,(f,g))
    @test fy < f
    @test attempts[] > 2
    @test y.Q+y.Q'+y.R'*y.R ≈ zeros(1,1)
    safeguarded = SafeguardedLineSearch(max_relative_step=0.05)
    y,fy,gy,v,alpha,nfg = safeguarded(objective,s,direction,(f,g))
    @test fy < f
    @test norm(alpha*direction) <= 0.05
end

@testset "Threaded observable execution" begin
    s = RandomState(3;seed=24,scaling=.2)
    r,k,rho = copy(s.R),copy(s.K),copy(s.ρ0)
    for objective in ((s;kw...)->fg_phi4(s,2.;kw...),
                      (s;kw...)->fg_sgordon(s,.5im;kw...),
                      (s;kw...)->fg_phi4_massmod(s,2.,.8;kw...))
        e,g = objective(s)
        for parallel in (:terms,:trajectories,:both)
            ep,gp = objective(s;parallel)
            @test ep == e
            @test gp == g
            @test objective(s;parallel,gradient=false) ≈ e atol=1e-10
        end
    end
    @test s.R == r && s.K == k && s.ρ0 == rho
    w = randn(MersenneTwister(25),ComplexF64,3,3)
    w /= norm(w)
    directional_check(s,(s;kw...)->fg_phi4(s,2.;parallel=:both,kw...),w)
    @test_throws ArgumentError fg_phi4(s,2.;parallel=:invalid)
    s.model_param.name = "phi4"
    s.model_param.coupling = 2.
    a = optimize_state(s;n_iter=3,verbosity=0,diagnostics=true)
    b = optimize_state(s;n_iter=3,verbosity=0,diagnostics=true,parallel=:both)
    @test a.state.R == b.state.R
    @test a.energy == b.energy
    s.integrator_param.maxiters = 1
    for parallel in (:terms,:trajectories,:both)
        @test_throws SolverFailure fg_phi4(s,2.;parallel)
    end
    if Threads.nthreads() > 1
        completed = Threads.Atomic{Int}(0)
        @test_throws DomainError RcmpsCore.evaluation_map(1:3,true) do i
            i == 1 && throw(DomainError(i))
            yield()
            Threads.atomic_add!(completed,1)
            i
        end
        @test completed[] == 2 # workers have all finished when the error reaches us
    end
end

@testset "Infinite-line kinetic resolvents" begin
    coherent = State(reshape(ComplexF64[.3+.1im],1,1),zeros(1,1))
    @test kinetic1(coherent;kinetic_backend=:resolvent,gradient=false) ≈ .1 atol=1e-14
    @test kinetic2(coherent;kinetic_backend=:resolvent,gradient=false) == 0
    @test norm(kinetic2(coherent;kinetic_backend=:resolvent)[2]) == 0
    s = RandomState(3;seed=90,scaling=.2)
    s.integrator_param.max_point=30.
    s.integrator_param.atol=1e-13;s.integrator_param.rtol=1e-12
    s.integrator_param.n_quad=256
    w=randn(MersenneTwister(91),ComplexF64,3,3);w/=norm(w)
    for objective in (kinetic1,kinetic2,(s;kw...)->fg_phi4(s,2.;kw...),
                       (s;kw...)->fg_phi4_massmod(s,2.,.8;kw...),
                       (s;kw...)->fg_sgordon(s,.5im;kw...))
        e,g = objective(s)
        er,gr = objective(s;kinetic_backend=:resolvent)
        @test er ≈ e atol=1e-10
        @test gr ≈ g atol=2e-8 rtol=2e-8
        et,gt = objective(s;kinetic_backend=:resolvent,parallel=:both)
        @test et == er && gt == gr
        directional_check(s,(s;kw...)->objective(s;kinetic_backend=:resolvent,kw...),w)
    end
    bad=KineticOptions(minnodes=2,maxnodes=4,atol=1e-15,rtol=0.)
    @test_throws SolverFailure kinetic2(s;kinetic_backend=:resolvent,kinetic_options=bad)
    @test_throws ArgumentError kinetic1(s;kinetic_backend=:invalid)
    @test_throws ArgumentError kinetic1(s;kinetic_backend=:resolvent,kinetic_options=KineticOptions(atol=Inf))
    @test_throws ArgumentError kinetic1(s;kinetic_backend=:resolvent,kinetic_options=KineticOptions(minnodes=1))
    s.model_param.name="phi4";s.model_param.coupling=2.
    result=optimize_state(s;n_iter=3,verbosity=0,diagnostics=true,
                          parallel=:both,kinetic_backend=:resolvent)
    @test result.energy<fg_phi4(s,2.;kinetic_backend=:resolvent,gradient=false)
end

include("kinetic_krylov.jl")
include("kinetic_plateau.jl")
include("excitations.jl")
include("excitation_action.jl")
include("excitation_eigsolve.jl")
