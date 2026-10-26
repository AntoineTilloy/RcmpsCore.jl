# Julia references for the optimizer of rcmps_gpu on phi4 states without Z2 symmetry
# (`rcmps-gpu-optimize --symmetry none`): energies, gradients and the one-point function
# <phi> = phi_monomial(s, 1) of RcmpsCore at the integration settings of the GPU optimizer,
# written as portable state directories (format 1, gpu/src/rcmps_gpu/io.py) that the GPU
# side reads (`--initial`, qualification/optimizer/check_nonz2.py).
#
# Usage, from the repository root (Julia threads for parallel=:both):
#   julia --project=. -t 8 gpu/qualification/optimizer/nonz2_references.jl evaluate NEW_DIR D SEED COUPLING
#   julia --project=. -t 8 gpu/qualification/optimizer/nonz2_references.jl optimize NEW_DIR D SEED COUPLING ITERATIONS [GRADTOL]
#   julia --project=. -t 8 gpu/qualification/optimizer/nonz2_references.jl trajectory NEW_DIR D SEED COUPLING ITERATIONS
#   julia --project=. -t 8 gpu/qualification/optimizer/nonz2_references.jl reevaluate NEW_DIR PORTABLE_DIR...
#
# evaluate: the random state RandomState(D; seed, scaling=1/sqrt(D)) at COUPLING.
# optimize: optimize_state (L-BFGS, history 100) from that state, with the kinetic angular
#   grid fixed at 512 nodes as the GPU optimizer, to GRADTOL (default 1e-7); NEW_DIR/state is
#   the final state, evaluated as below, and NEW_DIR/history.csv the OptimKit history.
# trajectory: the same for ITERATIONS iterations, every iterate stored (NEW_DIR/it000, ...).
# reevaluate: the states of portable directories (an optimized_state/ of a GPU run, say).
#
# Every evaluated state directory holds R, K, rho, the energy and gradient of the "fixed
# 512" evaluation (kinetic angular grid of 512 nodes: the discretization of the GPU optimizer)
# as `energy`/gradient.bin, phi = phi_monomial(s, 1), symmetry = "none", and as extra keys and
# files: julia_energy_1024 and gradient_1024.bin (the kinetic grid of 1024 nodes, to measure
# the angular discretization error of 512), julia_phi_closed (c(W) tr(R rho + rho R'), c(W)
# the integral of the source over the window).
@isdefined(PORTABLE_FORMAT) || include(joinpath(@__DIR__, "..", "..", "julia", "portable.jl"))
using Random, Printf
import OptimKit

const INTEGRATION = (n_quad=512, max_point=30.0, atol=1e-13, rtol=1e-12)
const KRYLOV = KineticKrylovOptions(tolerance=1e-12, maxdim=1024)
# The GPU optimizer's fixed grid: 256 then 512 nodes, the 512-node result accepted.
const FIXED512 = KineticOptions(minnodes=256, maxnodes=512, atol=1.0, rtol=1.0)
const FIXED1024 = KineticOptions(minnodes=512, maxnodes=1024, atol=1.0, rtol=1.0)

function prepare!(s::State, coupling)
    s.model_param = ModelParameters(name="phi4", coupling=coupling)
    s.integrator_param = IntegratorParameters(; INTEGRATION...)
    s.manifold_param = ManifoldParameters()
    return s
end
random_state(d, seed, coupling) = prepare!(RandomState(d; seed, scaling=1/sqrt(d)), coupling)

fg(s, options) = fg_phi4(s, real(s.model_param.coupling); parallel=:both, kinetic_backend=:krylov,
                         kinetic_options=options, krylov_options=KRYLOV)

"""c(W) = int_{-W}^{W} J(x) dx = 2 int_0^{sqrt W} transformed_source(t) dt (Gauss-Legendre)."""
function source_integral(max_point)
    a = sqrt(max_point)
    n = 400
    k = collect(1:n-1)
    f = eigen(SymTridiagonal(zeros(n), k ./ sqrt.(4 .* k.^2 .- 1)))
    x, w = a/2 .* (f.values .+ 1), a .* abs2.(f.vectors[1, :])
    # The integrand is smooth on (0, a]: K_{1/4}(t^2) t^(1/2) is analytic in t^2 near 0.
    return 2sum(w .* RcmpsCore.transformed_source.(x))
end

function evaluated(s::State, directory; source, extra=Dict{String,Any}())
    t0 = time()
    e, g = fg(s, FIXED512)
    t1 = time()
    ec, gc = fg(s, FIXED1024)
    t2 = time()
    phi = phi_monomial(s, 1; gradient=false)
    closed = source_integral(s.integrator_param.max_point) * real(tr(s.R*s.ρ0 + s.ρ0*s.R'))
    fields(x) = (; (key => getfield(x, key) for key in fieldnames(typeof(x)))...)
    metadata = Dict{String,Any}("julia_energy_1024" => ec,
        "julia_gradient_1024_metric_difference" => sqrt(metric(s, gc - g, gc - g)),
        "julia_phi_closed" => closed,
        "julia_kinetic" => "krylov, tolerance 1e-12, maxdim 1024; energy/gradient.bin: 512 angular nodes; " *
                           "*_1024: 1024 nodes",
        "julia_seconds_512" => t1 - t0, "julia_seconds_1024" => t2 - t1,
        "julia_threads" => Threads.nthreads())
    merge!(metadata, extra)
    write_portable(directory; R=s.R, K=s.K, rho=s.ρ0, gradient=g, model="phi4",
                   coupling=real(s.model_param.coupling), energy=e, gradient_norm=sqrt(metric(s, g, g)),
                   symmetry="none", phi,
                   producer="RcmpsCore $(pkgversion(RcmpsCore)), gpu/qualification/optimizer/nonz2_references.jl",
                   source=source, integration=fields(s.integrator_param), manifold=fields(s.manifold_param),
                   extra=metadata)
    open(joinpath(directory, "gradient_1024.bin"), "w") do io
        write(io, Matrix{ComplexF64}(gc))
    end
    @printf("%s: D=%d E=%.15f |g|=%.3e E_1024-E=%.2e phi=%.15f phi_closed-phi=%.2e (%.1f s, %.1f s)\n",
            directory, s.D, e, sqrt(metric(s, g, g)), ec - e, phi, closed - phi, t1 - t0, t2 - t1)
    flush(stdout)
    return e, g
end

function optimization(s::State, iterations, gradtol; record=nothing)
    objective(x) = fg(x, FIXED512)
    history = Tuple{Int,Float64,Float64,Float64}[]
    start = time()
    function callback(x, f, g, iteration)
        finalize!(x, f, g, iteration)
        gn = sqrt(metric(x, g, g))
        push!(history, (iteration, f, gn, time() - start))
        @printf("it=%d E=%.15f |g|=%.4e t=%.1f s\n", iteration, f, gn, time() - start)
        flush(stdout)
        isnothing(record) || record(x, f, g, iteration)
        return x, f, g
    end
    isnothing(record) || record(s, objective(s)..., 0)
    out, f, g, numfg, _ = OptimKit.optimize(objective, deepcopy(s),
        OptimKit.LBFGS(100; maxiter=iterations, gradtol, verbosity=1, linesearch=ls_default);
        inner=metric, retract=retraction, precondition=precondition, transport! = naive_transport!,
        finalize! = callback, isometrictransport=false)
    out.e0, out.grad = f, Matrix(g)
    return out, history, numfg
end

function write_history(path, history)
    open(path, "w") do io
        println(io, "iteration,energy,gradient_norm,seconds")
        for row in history
            println(io, join(row, ","))
        end
    end
end

function main(arguments)
    command = arguments[1]
    directory = arguments[2]
    ispath(directory) && error("Refusing to overwrite $directory: choose a new directory")
    if command == "reevaluate"
        mkpath(directory)
        for (i, path) in enumerate(arguments[3:end])
            if endswith(path, ".jld2")
                s = rcmps_load(path)
                p = (coupling=real(s.model_param.coupling), energy=s.e0)
            else
                p = read_portable(path)
                s = isnothing(p.rho) ? State(p.R, p.K) : State(p.R, p.K; ρ0=p.rho)
            end
            prepare!(s, p.coupling)
            evaluated(s, joinpath(directory, @sprintf("state%02d", i)); source=basename(normpath(path)),
                      extra=Dict{String,Any}("reevaluated_path" => abspath(path),
                                             "reevaluated_energy" => something(p.energy, NaN)))
        end
        return
    end
    d, seed, coupling = parse(Int, arguments[3]), parse(Int, arguments[4]), parse(Float64, arguments[5])
    s = random_state(d, seed, coupling)
    source = "RandomState($d; seed=$seed, scaling=1/sqrt($d))"
    seeds = Dict{String,Any}("seed" => seed)
    if command == "evaluate"
        evaluated(s, directory; source, extra=seeds)
    elseif command == "optimize"
        iterations = parse(Int, arguments[6])
        gradtol = length(arguments) >= 7 ? parse(Float64, arguments[7]) : 1e-7
        mkpath(directory)
        evaluated(s, joinpath(directory, "initial"); source, extra=seeds)
        out, history, numfg = optimization(s, iterations, gradtol)
        write_history(joinpath(directory, "history.csv"), history)
        rcmps_store(out, joinpath(directory, "state.jld2"); verbose=false)
        evaluated(out, joinpath(directory, "state"); source="optimize_state from $source, $(length(history)) iterations",
                  extra=merge(seeds, Dict{String,Any}("iterations" => length(history), "evaluations" => numfg)))
    elseif command == "trajectory"
        iterations = parse(Int, arguments[6])
        mkpath(directory)
        out, history, _ = optimization(s, iterations, 1e-12; record=(x, f, g, iteration) -> begin
            fields(y) = (; (key => getfield(y, key) for key in fieldnames(typeof(y)))...)
            write_portable(joinpath(directory, @sprintf("it%03d", iteration)); R=x.R, K=x.K, rho=x.ρ0, gradient=g,
                model="phi4", coupling=coupling, energy=f, gradient_norm=sqrt(metric(x, g, g)),
                producer="RcmpsCore $(pkgversion(RcmpsCore)), gpu/qualification/optimizer/nonz2_references.jl",
                source="iterate $iteration of optimize_state from $source", integration=fields(x.integrator_param),
                manifold=fields(x.manifold_param), symmetry="none", extra=Dict{String,Any}("iteration" => iteration))
        end)
        write_history(joinpath(directory, "history.csv"), history)
    else
        error("unknown command $command")
    end
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && main(ARGS)
