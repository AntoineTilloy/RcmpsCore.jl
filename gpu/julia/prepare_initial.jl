# A random initial state for the GPU optimizer (rcmps-gpu-optimize --initial), reproducible
# in Julia: the matrices of RandomStateZ2(D; seed, scaling), or with --symmetry none those of
# RandomState(D; seed, scaling) (MersenneTwister draw order), with the stationary density
# solved once at the tighter tolerance 1e-13 (method :cptp). Written as a portable state
# directory, format 1, for phi4 at the given coupling, with the integration settings of the GPU
# optimizer's defaults (window 30, 512 nodes, 1e-13/1e-12), the default manifold parameters and
# the symmetry ("Z2" or "none"). Nothing is evaluated: no energy, gradient or optimization.
#
# Usage: julia --project=PATH/TO/RcmpsCore.jl prepare_initial.jl [--symmetry z2|none] D SEED COUPLING NEW_DIRECTORY [SCALING]
# (SCALING defaults to 1/sqrt(D); --symmetry z2, the default, needs an even D)
@isdefined(PORTABLE_FORMAT) || include(joinpath(@__DIR__, "portable.jl"))   # once, when scripts include each other
using Random

const PREPARE_STATIONARY_TOLERANCE = 1e-13

"""R and K of RandomStateZ2(d; seed, scaling), drawn in its order (src/state.jl)."""
function random_z2_parameters(d::Integer, seed::Integer, scaling::Real)
    d > 0 && iseven(d) || throw(ArgumentError("Z2 initialization requires a positive even D"))
    isfinite(scaling) && scaling > 0 || throw(ArgumentError("scaling must be finite and positive"))
    rng = MersenneTwister(seed)
    n = d ÷ 2
    k1, k2 = randn(rng, ComplexF64, n, n), randn(rng, ComplexF64, n, n)
    z = zeros(ComplexF64, n, n)
    r = [z randn(rng, ComplexF64, n, n); randn(rng, ComplexF64, n, n) z]
    return scaling * r, scaling * [k1+k1' z; z k2+k2']
end

"""R and K of RandomState(d; seed, scaling), drawn in its order (src/state.jl): K first."""
function random_parameters(d::Integer, seed::Integer, scaling::Real)
    d > 0 || throw(ArgumentError("D must be positive"))
    isfinite(scaling) && scaling > 0 || throw(ArgumentError("scaling must be finite and positive"))
    rng = MersenneTwister(seed)
    k = randn(rng, ComplexF64, d, d)
    return scaling * randn(rng, ComplexF64, d, d), scaling * (k + k')
end

function prepare_initial(d::Integer, seed::Integer, coupling::Real, directory::AbstractString;
                         scaling::Real=1/sqrt(d), symmetry="Z2", check=true)
    ispath(directory) && error("Refusing to overwrite $directory: choose a new directory")
    isfinite(coupling) && coupling >= 0 || throw(ArgumentError("phi4 coupling must be finite and nonnegative"))
    symmetry in ("Z2", "none") || throw(ArgumentError("symmetry must be \"Z2\" or \"none\", not $(repr(symmetry))"))
    z2 = symmetry == "Z2"
    r, k = (z2 ? random_z2_parameters : random_parameters)(d, seed, scaling)
    if check   # the same draws as RandomStateZ2 or RandomState (which also solve for rho, at their default tolerance)
        t = (z2 ? RandomStateZ2 : RandomState)(d; seed, scaling)
        t.R == r && t.K == (k + k') / 2 || error("the draws differ from $(z2 ? "RandomStateZ2" : "RandomState")")
    end
    q = -im * k - r' * r / 2
    stationary = steady_state(q, r; method=:cptp, tol=PREPARE_STATIONARY_TOLERANCE)
    s = State(r, k; ρ0=stationary.rho, steady_tol=PREPARE_STATIONARY_TOLERANCE)
    fields(x) = (; (key => getfield(x, key) for key in fieldnames(typeof(x)))...)
    write_portable(directory; R=s.R, K=s.K, rho=s.ρ0, model="phi4", coupling=Float64(coupling),
                   producer="RcmpsCore $(pkgversion(RcmpsCore)), gpu/julia/prepare_initial.jl",
                   source="$(z2 ? "RandomStateZ2" : "RandomState")($d; seed=$seed, scaling=$scaling)", symmetry,
                   integration=(n_quad=512, max_point=30.0, atol=1e-13, rtol=1e-12),
                   manifold=fields(ManifoldParameters()),
                   extra=Dict{String,Any}("seed"=>Int(seed), "scaling"=>Float64(scaling),
                       "random_generator"=>"Julia Random.MersenneTwister; randn(ComplexF64); " *
                                           "$(z2 ? "RandomStateZ2" : "RandomState") draw order",
                       "stationary_method"=>string(stationary.method), "stationary_tolerance"=>PREPARE_STATIONARY_TOLERANCE,
                       "stationary_iterations"=>stationary.iterations, "stationary_residual"=>stationary.residual,
                       "rho_eigmin"=>eigmin(Hermitian(s.ρ0))))
    return s, stationary
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    usage = "Usage: julia --project=PATH/TO/RcmpsCore.jl prepare_initial.jl [--symmetry z2|none] D SEED COUPLING NEW_DIRECTORY [SCALING]"
    arguments, symmetry = copy(ARGS), "Z2"
    if !isempty(arguments) && arguments[1] == "--symmetry"
        length(arguments) >= 2 && arguments[2] in ("z2", "none") || error(usage)
        symmetry = arguments[2] == "z2" ? "Z2" : "none"
        arguments = arguments[3:end]
    end
    length(arguments) in (4, 5) || error(usage)
    d, seed, coupling = parse(Int, arguments[1]), parse(Int, arguments[2]), parse(Float64, arguments[3])
    scaling = length(arguments) == 5 ? parse(Float64, arguments[5]) : 1/sqrt(d)
    s, stationary = prepare_initial(d, seed, coupling, arguments[4]; scaling, symmetry)
    println("Prepared D=$d seed=$seed g=$coupling symmetry=$symmetry: stationary residual $(stationary.residual), " *
            "rho min $(eigmin(Hermitian(s.ρ0))) in $(arguments[4])")
end
