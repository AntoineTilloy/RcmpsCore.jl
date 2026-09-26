# From the repository root:
# JULIA_LOAD_PATH='@:@stdlib' julia --startup-file=no --project=. examples/excitations.jl
# Optional first argument: a refactor .jld2 state with matching model metadata.
# Optional second argument: number of retained branches (default 1).
using RcmpsCore, LinearAlgebra
BLAS.set_num_threads(1)

if isempty(ARGS)
    s = State(zeros(1,1),zeros(1,1))
    s.model_param.name = "phi4"
    s.model_param.coupling = 0.
    println("Free-vacuum check: exact E(p)=sqrt(1+p^2), bare mass and c are one.")
else
    s = rcmps_load(ARGS[1])
    println("Loaded D=$(s.D), model=$(s.model_param.name), coupling=$(s.model_param.coupling).")
    println("Check ground-state convergence, bond dimension, and source-window convergence.")
end

nev = length(ARGS)>=2 ? parse(Int,ARGS[2]) : 1
momenta = collect(-2.:.5:2.)
dispersion = dispersion_relation(s,momenta;nev)
println("momentum,"*join(["energy_$i" for i in 1:nev],","))
for (i,p) in enumerate(momenta)
    println(join(vcat(p,dispersion.energies[i,:]),","))
end
println("Rest energy in represented sector: ",dispersion.energies[findfirst(iszero,momenta),1])
println("Relativistic fit for energy-sorted branch 1: ",relativistic_diagnostics(dispersion))
println("Largest finite-matrix residual: ",maximum(maximum(s.residuals) for s in dispersion.spectra))
