@testset "Reverse-contracted excitation Hamiltonian action" begin
    s=RandomState(2;seed=42,scaling=.5)
    s.integrator_param.max_point=24
    s.integrator_param.n_quad=192
    s.integrator_param.atol=1e-12;s.integrator_param.rtol=1e-12
    u=randn(MersenneTwister(1),ComplexF64,2,2)
    w=randn(MersenneTwister(2),ComplexF64,2,2)
    for p in (0.,.7), model in (ExcitationModel(:phi4,1.),ExcitationModel(:shg,.6),ExcitationModel(:sg,.6))
        a=excitation_hamiltonian_action(s,p,w;model)
        exact=excitation_matrix_element(s,p,u,w;model)
        @test dot(u,a) ≈ exact atol=2e-8 rtol=2e-9
        @test excitation_hamiltonian_action(s,p,(1+im)*w;model) ≈ (1+im)*a atol=2e-8
    end
    model=ExcitationModel(:phi4,1.)
    @test excitation_hamiltonian_action(s,.7,w;model,parallel=:terms) ≈
        excitation_hamiltonian_action(s,.7,w;model) atol=1e-12
    for parallel in (:trajectories,:both)
        @test excitation_hamiltonian_action(s,.7,w;model,parallel) ≈
            excitation_hamiltonian_action(s,.7,w;model) atol=1e-12
    end
    @test excitation_hamiltonian_action(s,.7,w;model,kinetic_backend=:ode) ≈
        excitation_hamiltonian_action(s,.7,w;model) atol=2e-8
    vacuum=State(zeros(1,1),zeros(1,1));w1=ones(ComplexF64,1,1)
    @test excitation_hamiltonian_action(vacuum,.7,w1;model=ExcitationModel(:phi4,0.)) ≈ sqrt(1+.7^2)*w1 atol=1e-11
    @test_throws DimensionMismatch excitation_hamiltonian_action(s,0.,w1;model)
    @test_throws ArgumentError excitation_hamiltonian_action(s,0.,w;model,kinetic_backend=:invalid)
    direct=excitation_spectrum(s,.7;model,nev=2,assembly=:action)
    bilinear=excitation_spectrum(s,.7;model,nev=2,assembly=:bilinear)
    @test direct.hamiltonian ≈ bilinear.hamiltonian atol=2e-8
    @test direct.energies ≈ bilinear.energies atol=2e-8
    @test_throws ArgumentError excitation_hamiltonian(s,0.;model,assembly=:invalid)
    @test_throws ArgumentError excitation_hamiltonian_action(s,0.,w;model,parallel=:invalid)
end
