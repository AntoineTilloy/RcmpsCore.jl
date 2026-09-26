# D=16 phi4 g=2 mass-gap regression for the iterative excitation solver.
# The Z2 ground state test/fixtures/phi4_g2_D16_z2_state.jld2 is the mass-scan input
# research/mass_scan/inputs/D16/g2.0000/state.jld2 of the companion research repository
# rcmps-phi4-z2 (sha256 a7b04347...); its recorded validated mass is 0.3523798021192911.
# Run from the repository root (8 min with two threads on a loaded 12-core workstation):
# JULIA_LOAD_PATH='@:@stdlib' julia --project=. --threads=2 benchmark/excitation_mass_d16.jl
using RcmpsCore, LinearAlgebra
BLAS.set_num_threads(1)
const RC = RcmpsCore
source = joinpath(@__DIR__,"..","test","fixtures","phi4_g2_D16_z2_state.jld2")
s = rcmps_load(source)
s.integrator_param.max_point=30.; s.integrator_param.n_quad=512
s.integrator_param.atol=1e-13; s.integrator_param.rtol=1e-12
model = ExcitationModel(:phi4,2.0)
refined = IntegratorParameters(max_point=40.,n_quad=768,atol=1e-15,rtol=1e-14)
recorded = 0.3523798021192911
println("threads=",Threads.nthreads()," D=",s.D," parity_error=",RC.parity_error(s,excitation_parity(s)))
flush(stdout)
t = time()
# Library defaults: odd sector, density gauge, fresh-restart Davidson.
r = excitation_eigsolve(s,0.0;model,refined,refined_kinetic_nodes=512,gauge=:density)
mass = r.validated_energies[1]
difference = mass-recorded
println((converged=r.converged,accepted=r.accepted,stop=r.stop_reason,numops=r.numops,
         action_calls=r.action_calls,solver_residual=r.residuals[1],seconds=time()-t))
println(r.validation[1])
println("validated mass = ",mass)
println("recorded mass  = ",recorded)
println("difference     = ",difference)
flush(stdout)
abs(difference) <= 1e-9 || error("D16 mass regression failed: |difference| = $(abs(difference)) > 1e-9")
println("D16 mass regression passed")
