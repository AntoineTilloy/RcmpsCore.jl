# From the repository root:
# JULIA_LOAD_PATH='@:@stdlib' julia --startup-file=no --threads=4 --project=. examples/excitations.jl
# Optional first argument: a refactor .jld2 state with matching model metadata.
# Optional second argument: number of retained branches (default 1).
# Loaded states use the iterative solver (odd sector for Z2 states) and the
# state's integration settings; set a converged window, n_quad and tolerances first.
using RcmpsCore, LinearAlgebra
BLAS.set_num_threads(1)

if isempty(ARGS)
    s = State(zeros(1,1),zeros(1,1))
    s.model_param.name = "phi4"
    s.model_param.coupling = 0.
    solver = :dense
    momenta = collect(-2.:.5:2.)
    println("Free-vacuum check: exact E(p)=sqrt(1+p^2), bare mass and c are one.")
else
    s = rcmps_load(ARGS[1])
    solver = :iterative
    momenta = collect(0.:.25:1.)
    println("Loaded D=$(s.D), model=$(s.model_param.name), coupling=$(s.model_param.coupling).")
    println("Z2 parity: ",isnothing(excitation_parity(s)) ? "none (full tangent space)" : "found (odd sector)")
    println("Check ground-state convergence, bond dimension, and source-window convergence.")
end

nev = length(ARGS)>=2 ? parse(Int,ARGS[2]) : 1
dispersion = dispersion_relation(s,momenta;nev,solver)
println("momentum,"*join(["energy_$i" for i in 1:nev],","))
for (i,p) in enumerate(momenta)
    println(join(vcat(p,dispersion.energies[i,:]),","))
end
println("Rest energy in represented sector: ",dispersion.energies[findfirst(iszero,momenta),1])
println("Relativistic fit for energy-sorted branch 1: ",relativistic_diagnostics(dispersion))
if solver == :dense
    println("Largest finite-matrix residual: ",maximum(maximum(r.residuals) for r in dispersion.spectra))
else
    checks = [v for r in dispersion.spectra for v in r.validation]
    println("Largest fresh residual: ",maximum(max(v.standard_residual,v.refined_residual) for v in checks))
    println("Largest energy spread across checks: ",maximum(v.energy_spread for v in checks))
    println("All searches converged: ",all(r.converged for r in dispersion.spectra))
end
