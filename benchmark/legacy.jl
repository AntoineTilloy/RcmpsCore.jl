# Execute the original (v0.1.0) sources unchanged except the OrdinaryDiffEq umbrella
# import: the same Vern7 algorithm now comes from its dedicated package; SciML 3 requires
# sol.u[i] for saved-state indexing. All old kernels,
# quadrature, caches, and fixed-point routines are preserved, including Octavian.
# The legacy files are read from the git tag v0.1.0 of this repository
# (`git show v0.1.0:src/<file>`). Set ENV["RCMPS_LEGACY_SRC"] to a directory holding
# the legacy src/ files to read them from there instead.
module Legacy
const repo = normpath(joinpath(@__DIR__,".."))
const override = get(ENV,"RCMPS_LEGACY_SRC","")
for name in ("rcmps_structure.jl","helper_funcs.jl","ode_generators_noalloc.jl",
             "expectation_values.jl","riemannian_optimization.jl")
    if isempty(override)
        source = read(`git -C $(repo) show v0.1.0:src/$(name)`,String)
        filename = "v0.1.0:src/$(name)"
    else
        filename = joinpath(override,name)
        source = read(filename,String)
    end
    source = replace(source,"using OrdinaryDiffEq" => "using SciMLBase: ODEProblem, solve\nusing OrdinaryDiffEqVerner: Vern7")
    source = replace(source,"ρ[i]"=>"ρ.u[i]","σ[end-i+1]"=>"σ.u[end-i+1]")
    include_string(@__MODULE__,source,filename)
end
end
