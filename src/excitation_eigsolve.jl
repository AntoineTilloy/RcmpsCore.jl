# Parity sectors, white tangent coordinates, and the iterative lowest-level solver.
# See docs/excitations.tex: parity sectors, white coordinates, iterative calculation.

# Relative violation of Z R Z = -R, Z K Z = K and Z rho Z = rho for Z = Diagonal(z).
function parity_error(s::State,z)
    zz = z*z'
    return max(norm(zz.*s.R+s.R)/max(1,norm(s.R)),norm(zz.*s.K-s.K)/max(1,norm(s.K)),
               norm(zz.*s.ρ0-s.ρ0))
end

"""Diagonal auxiliary Z₂ parity of `s`, as a ±1 vector, or `nothing`.
`:auto` recognizes the balanced block layout of `RandomStateZ2`, Z = diag(1,…,1,-1,…,-1),
when R is odd and K and ρ are even to relative tolerance `tol`. A supplied ±1 vector is
checked with the same tolerance. For another unitary Z, rotate the state first.
"""
function excitation_parity(s::State,parity=:auto;tol=1e-12)
    isnothing(parity) && return nothing
    if parity === :auto
        iseven(s.D) || return nothing
        z = vcat(ones(Int,s.D÷2),-ones(Int,s.D÷2))
        return parity_error(s,z) <= tol ? z : nothing
    end
    parity isa AbstractVector || throw(ArgumentError("parity must be :auto, nothing, or a ±1 vector"))
    z = Int.(collect(parity))
    length(z) == s.D && all(in((-1,1)),z) && any(==(1),z) && any(==(-1),z) ||
        throw(ArgumentError("parity must be a ±1 vector of length D with both signs"))
    violation = parity_error(s,z)
    violation <= tol || throw(ArgumentError("state is not Z2 symmetric under the supplied parity (error $violation)"))
    return z
end

"""
    TangentCoordinates(s; sector=:auto, parity=:auto, gauge=:density, metric_tol=0.0)

Orthonormal "white" coordinates for gauge-fixed tangents (V=-R†W) in one sector:
a coefficient vector c maps to W = X B†, with X supported on `indices` and B†ρB = 1
on the resolved metric support, so that tr(W ρ W†) = ‖c‖².

For a Z₂-symmetric state (see `excitation_parity`), physical parity acts as
W ↦ -ZWZ: block-diagonal W form the `:odd` sector and off-diagonal W the `:even`
sector, each with D²/2 directions for balanced blocks. `:all` keeps every direction.
`sector=:auto` selects `:odd` when a parity is found and `:all` otherwise.

`gauge=:density` rotates the auxiliary basis into the eigenbasis of ρ within each
parity block (an exact unitary gauge change) and whitens the stored density by a
Cholesky factor; this was needed for Float64 accuracy at D≥56. `gauge=:original`
keeps the gauge of `s`. Eigenvalues of ρ at or below
`max(metric_tol, 10D·eps)*maximum(ρ)` are dropped: below the Float64 resolution of a
Hermitian eigensolver they are roundoff, not physical directions. Cholesky whitening is
used when none is dropped. `density_condition` reports λmax/λmin of the kept spectrum.
Coordinates and all internal environments use `coordinates.state`; convert
tangents with `original_tangent` and `coordinate_tangent`.
"""
struct TangentCoordinates
    state::State
    rotation::Matrix{ComplexF64}
    whitening::Matrix{ComplexF64}
    indices::Vector{CartesianIndex{2}}
    parity::Union{Nothing,Vector{Int}}
    parity_error::Float64
    sector::Symbol
    gauge::Symbol
    metric_rank::Int
    discarded_metric_weight::Float64
    metric_defect::Float64
    density_condition::Float64
end

function TangentCoordinates(s::State;sector=:auto,parity=:auto,gauge=:density,metric_tol=0.0,
                            parity_tol=1e-12)
    isfinite(metric_tol) && 0<=metric_tol<1 || throw(ArgumentError("metric_tol must lie in [0,1)"))
    gauge in (:density,:original) || throw(ArgumentError("gauge must be :density or :original"))
    z = excitation_parity(s,parity;tol=parity_tol)
    sector === :auto && (sector = isnothing(z) ? :all : :odd)
    sector in (:all,:odd,:even) || throw(ArgumentError("sector must be :auto, :all, :odd or :even"))
    sector != :all && isnothing(z) &&
        throw(ArgumentError("the $sector sector requires a Z2-symmetric state; supply its parity"))
    d = s.D
    blocks = isnothing(z) ? [collect(1:d)] : [findall(==(1),z),findall(==(-1),z)]
    # Block eigenvectors of rho: off-block elements of rho are zero by parity.
    u = zeros(ComplexF64,d,d);lambda = zeros(d)
    for b in blocks
        f = eigen(Hermitian(s.ρ0[b,b]))
        u[b,b] = f.vectors;lambda[b] = f.values
    end
    cutoff = max(metric_tol,10d*eps())*maximum(lambda)
    resolved(values) = values .> cutoff
    keep = resolved(lambda)
    any(keep) || throw(SolverFailure("empty tangent metric support"))
    if gauge == :density
        rho = u'*s.ρ0*u
        t = State(u'*s.R*u,u'*s.K*u;ρ0=(rho+rho')/2)
        copy_param!(t,s);t.e0 = s.e0
        rotation = u
    else
        t,rotation = s,Matrix{ComplexF64}(I,d,d)
    end
    # Column j of B lies in the parity block of j. Cholesky keeps every column;
    # eigen-whitening keeps the resolved eigen-directions.
    b = zeros(ComplexF64,d,d);colkeep = trues(d)
    for blk in blocks
        f = all(keep) ? cholesky(Hermitian(t.ρ0[blk,blk]);check=false) : nothing
        if !isnothing(f) && issuccess(f)
            b[blk,blk] = inv(f.U)
        else
            f = gauge == :density ? eigen(Hermitian(t.ρ0[blk,blk])) : (values=lambda[blk],vectors=u[blk,blk])
            mask = resolved(f.values)
            b[blk,blk] = f.vectors*Diagonal(mask./sqrt.(max.(f.values,floatmin())))
            colkeep[blk] = mask
        end
    end
    columns = findall(colkeep)
    allowed(i,j) = sector == :all || (sector == :odd) == (z[i] == z[j])
    indices = [CartesianIndex(i,j) for j in columns for i in 1:d if allowed(i,j)]
    isempty(indices) && throw(SolverFailure("empty tangent sector"))
    defect = norm(b'*t.ρ0*b-Diagonal(in(columns).(1:d)))
    defect <= 1e-6 || throw(SolverFailure("tangent whitening is unresolved: metric defect $defect; use gauge=:density or a larger metric_tol"))
    return TangentCoordinates(t,rotation,b,indices,z,isnothing(z) ? NaN : parity_error(s,z),
                              sector,gauge,count(keep),sum(abs,lambda[.!keep];init=0.0),defect,
                              maximum(lambda)/minimum(lambda[keep]))
end

Base.length(c::TangentCoordinates) = length(c.indices)
"""Tangent W (in `c.state`'s gauge) of a white coefficient vector."""
function tangent_matrix(c::TangentCoordinates,x::AbstractVector)
    length(x) == length(c) || throw(DimensionMismatch("coordinate vector has the wrong length"))
    m = zeros(ComplexF64,c.state.D,c.state.D)
    m[c.indices] = x
    return m*c.whitening'
end
"""Metric-orthogonal projection of W (in `c.state`'s gauge) onto the coordinates."""
tangent_coordinates(c::TangentCoordinates,w::AbstractMatrix) = (w*c.state.ρ0*c.whitening)[c.indices]
"""Coordinates of a Hamiltonian action F: dot(x, covector(F(W))) == dot(tangent_matrix(x), F)."""
covector_coordinates(c::TangentCoordinates,f::AbstractMatrix) = (f*c.whitening)[c.indices]
original_tangent(c::TangentCoordinates,w::AbstractMatrix) = c.rotation*w*c.rotation'
coordinate_tangent(c::TangentCoordinates,w::AbstractMatrix) = c.rotation'*w*c.rotation
function tangent_basis(c::TangentCoordinates)
    n = length(c)
    return [tangent_matrix(c,(1:n) .== k) for k in 1:n]
end

"""Hermitian excitation operator in white coordinates, x ↦ [F(W(x)) B]_sector.
With fixed `kinetic_nodes` the kinetic part is exactly linear; adaptive ODE potential
terms are linear to their integration tolerance. `calls` counts applications.
"""
struct ExcitationOperator{C,M}
    context::C
    coordinates::TangentCoordinates
    model::ExcitationModel
    kinetic_backend::Symbol
    kinetic_nodes::Union{Nothing,Int}
    kinetic_options::KineticOptions
    method::M
    parallel::Symbol
    calls::Base.RefValue{Int}
end
function ExcitationOperator(ctx,coordinates,model;kinetic_backend=:resolvent,kinetic_nodes=256,
                            kinetic_options=KineticOptions(),method=Vern7(),parallel=:serial)
    ctx.state.D == coordinates.state.D || throw(DimensionMismatch("context and coordinates differ"))
    kinetic_backend in (:resolvent,:ode) || throw(ArgumentError("excitation kinetic backend must be :resolvent or :ode"))
    check_parallel(parallel)
    nodes = isnothing(kinetic_nodes) ? nothing : check_kinetic_nodes(kinetic_nodes)
    return ExcitationOperator(ctx,coordinates,model,kinetic_backend,nodes,kinetic_options,method,
                              parallel,Ref(0))
end
function (op::ExcitationOperator)(x::AbstractVector)
    f = excitation_hamiltonian_action(op.context,tangent_matrix(op.coordinates,x);model=op.model,
        kinetic_backend=op.kinetic_backend,kinetic_options=op.kinetic_options,
        kinetic_nodes=op.kinetic_nodes,method=op.method,parallel=op.parallel)
    op.calls[] += 1
    return covector_coordinates(op.coordinates,f)
end
with_context(op::ExcitationOperator,ctx;kinetic_nodes=op.kinetic_nodes) =
    ExcitationOperator(ctx,op.coordinates,op.model,op.kinetic_backend,kinetic_nodes,
                       op.kinetic_options,op.method,op.parallel,Ref(0))

"""Positive preconditioner from the coarse kinetic matrix K on a `nodes`-point θ grid:
r ↦ Σ vₖ⟨vₖ,r⟩/max(floor,κₖ). The floor defaults to the kinetic coefficient (the free
mass-one gap in model units). K costs one coarse kinetic action per coordinate.
"""
function kinetic_preconditioner(op::ExcitationOperator;nodes=8,floor=nothing,threaded=true)
    c,ctx = op.coordinates,op.context
    s,n = ctx.state,length(c)
    kinetics = filter(t->t[2].kinetic,op.model.terms)
    isempty(kinetics) && throw(ArgumentError("the kinetic preconditioner requires kinetic terms"))
    nodes = check_kinetic_nodes(nodes)
    columns = evaluation_map(1:n,threaded) do j
        w = tangent_matrix(c,(1:n) .== j)
        v = -s.R'*w
        rw = excitation_poisson(ctx,ket_insertion(s,v,w,s.ρ0);shift=1)
        covector_coordinates(c,excitation_kinetic_action_sum(ctx,w,v,rw,kinetics,nodes))
    end
    k = reduce(hcat,columns)
    all(isfinite,k) || throw(SolverFailure("nonfinite kinetic preconditioner"))
    f = eigen(Hermitian((k+k')/2))
    lower = isnothing(floor) ? maximum(first,kinetics) : Float64(floor)
    isfinite(lower) && lower > 0 || throw(ArgumentError("preconditioner floor must be positive"))
    apply(r) = f.vectors*((f.vectors'*r)./max.(lower,f.values))
    return (apply=apply,matrix=k,values=f.values,vectors=f.vectors,
            skew=norm(k-k')/max(1,norm(k)),floor=lower,nodes=nodes)
end

"""White coordinates of the gauge-fixed field insertion φ(0)|Ψ⟩ at the context momentum:
W = 1 + [R,X] with (L†+ip)X = R+R† (trace-projected). Only a start vector."""
function field_seed(op::ExcitationOperator)
    ctx = op.context;s = ctx.state
    x = -action_dual_poisson(ctx,s.R+s.R';shifted=true)
    return tangent_coordinates(op.coordinates,Matrix{ComplexF64}(I,s.D,s.D)+s.R*x-x*s.R)
end

function initial_vectors(op,start,nev,pre)
    c = op.coordinates;n = length(c)
    as_coordinates(x::AbstractMatrix) = (size(x) == (c.state.D,c.state.D) ||
        throw(DimensionMismatch("start tangent has the wrong size"));
        tangent_coordinates(c,coordinate_tangent(c,x)))
    as_coordinates(x::AbstractVector{<:Number}) = (length(x) == n ||
        throw(DimensionMismatch("start coordinates have the wrong length")); Vector{ComplexF64}(x))
    kinetic(k) = isnothing(pre) ? Vector{ComplexF64}[] : [pre.vectors[:,j] for j in 1:min(k,n)]
    if start isa Symbol
        start in (:auto,:field,:kinetic,:random) ||
            throw(ArgumentError("start must be :auto, :field, :kinetic, :random, coordinates or tangents"))
        if start in (:auto,:field)
            x = field_seed(op)
            # The field insertion has unit norm up to its gauge part; ~0 means another sector.
            norm(x) > 1e-6 && return vcat([x],kinetic(nev-1)),:field
            start == :field && throw(ArgumentError("the field insertion vanishes in the $(c.sector) sector"))
        end
        if start in (:auto,:kinetic) && !isnothing(pre)
            return kinetic(nev),:kinetic
        end
        start == :kinetic && throw(ArgumentError("start=:kinetic requires preconditioner=:kinetic"))
        rng = MersenneTwister(1234)
        return [randn(rng,ComplexF64,n) for _ in 1:nev],:random
    end
    starts = start isa AbstractMatrix || start isa AbstractVector{<:Number} ? [start] : collect(start)
    return [as_coordinates(x) for x in starts],:supplied
end

refined_integration(p::IntegratorParameters) =
    IntegratorParameters(n_quad=2p.n_quad,max_point=p.max_point+10,atol=p.atol/10,rtol=p.rtol/10,
                         maxiters=p.maxiters)

# One fresh action is a continuum Rayleigh quotient with an exact (for that action) residual.
function fresh_level(op,x)
    h = op(x);value = dot(x,h)/dot(x,x)
    return (energy=real(value),imaginary=imag(value),residual=norm(h-real(value)*x)/norm(x),action=h)
end
function bilinear_level(ctx,op,x;kinetic_nodes)
    w = tangent_matrix(op.coordinates,x)
    value = excitation_element(ExcitationPair(ctx,w,w),op.model;kinetic_backend=op.kinetic_backend,
        kinetic_options=op.kinetic_options,kinetic_nodes,method=op.method)/tangent_metric(ctx.state,w,w)
    return (energy=real(value),imaginary=imag(value))
end

const LEVEL_KEYWORDS = (:kinetic_options,:method,:callback)

start_list(start) = start isa AbstractMatrix || start isa AbstractVector{<:Number} ? [start] : collect(start)

# Validate the cheap request before any dense factorization; returns the Davidson options.
function check_levels_request(coordinates::TangentCoordinates;nev=1,davidson=nothing,start=:auto,
        preconditioner=:kinetic,refine_steps=0,kinetic_nodes=256,preconditioner_nodes=8,
        acceptance_residual=1e-5,validation_tol=1e-7,kinetic_backend=:resolvent,parallel=:both,
        validate=true,refined=nothing,refined_kinetic_nodes=nothing,kw...)
    n,d = length(coordinates),coordinates.state.D
    unknown = setdiff(keys(kw),LEVEL_KEYWORDS)
    isempty(unknown) || throw(ArgumentError("unknown excitation_eigsolve keywords: $(join(unknown,", "))"))
    kinetic_backend in (:resolvent,:ode) || throw(ArgumentError("excitation kinetic backend must be :resolvent or :ode"))
    check_parallel(parallel)
    nev isa Integer && 1 <= nev <= n ||
        throw(ArgumentError("nev=$nev must lie between 1 and the sector dimension $n"))
    options = something(davidson,DavidsonOptions(;maxspace=max(8,2nev+2)))
    check_davidson(options,nev)
    refine_steps isa Integer && refine_steps >= 0 && (refine_steps == 0 || nev == 1) ||
        throw(ArgumentError("refine_steps must be a nonnegative integer and requires nev=1"))
    preconditioner in (:kinetic,:none) || preconditioner isa Function ||
        throw(ArgumentError("preconditioner must be :kinetic, :none, or a function"))
    if start isa Symbol
        start in (:auto,:field,:kinetic,:random) ||
            throw(ArgumentError("start must be :auto, :field, :kinetic, :random, coordinates or tangents"))
        start === :kinetic && preconditioner !== :kinetic &&
            throw(ArgumentError("start=:kinetic requires preconditioner=:kinetic"))
    else
        starts = start_list(start)
        for x in starts
            x isa AbstractVector{<:Number} && length(x) == n || x isa AbstractMatrix && size(x) == (d,d) ||
                throw(DimensionMismatch("start entries must be length-$n coordinates or $d×$d tangents"))
        end
        1 <= length(starts) < options.maxspace ||
            throw(ArgumentError("between 1 and maxspace-1=$(options.maxspace-1) start vectors are accepted"))
    end
    validate isa Bool || throw(ArgumentError("validate must be true or false"))
    isnothing(refined) || refined isa IntegratorParameters && (quadrature(refined);true) ||
        throw(ArgumentError("refined must be nothing or IntegratorParameters"))
    isnothing(refined_kinetic_nodes) || check_kinetic_nodes(refined_kinetic_nodes)
    isnothing(kinetic_nodes) || check_kinetic_nodes(kinetic_nodes)
    preconditioner === :kinetic && check_kinetic_nodes(preconditioner_nodes)
    isfinite(acceptance_residual) && acceptance_residual > 0 && isfinite(validation_tol) && validation_tol > 0 ||
        throw(ArgumentError("acceptance_residual and validation_tol must be positive"))
    return options
end

function solve_levels(coordinates,ctx,model;nev::Integer=1,kinetic_backend=:resolvent,kinetic_nodes=256,
        kinetic_options=KineticOptions(),method=Vern7(),parallel=:both,preconditioner=:kinetic,
        preconditioner_nodes=8,start=:auto,davidson=nothing,
        validate=true,refined=nothing,refined_kinetic_nodes=nothing,refine_steps=0,
        validation_tol=1e-7,acceptance_residual=1e-5,callback=nothing,context_seconds=0.0)
    options = check_levels_request(coordinates;nev,davidson,start,preconditioner,refine_steps,
        kinetic_nodes,preconditioner_nodes,acceptance_residual,validation_tol,kinetic_backend,parallel,
        validate,refined,refined_kinetic_nodes)
    op = ExcitationOperator(ctx,coordinates,model;kinetic_backend,kinetic_nodes,kinetic_options,method,parallel)
    tick = time()
    pre = preconditioner === :kinetic ?
        kinetic_preconditioner(op;nodes=preconditioner_nodes,threaded=parallel != :serial) : nothing
    apply = preconditioner === :kinetic ? pre.apply : preconditioner === :none ? identity : preconditioner
    preconditioner_seconds = time()-tick
    starts,startkind = initial_vectors(op,start,nev,pre)
    tick = time()
    result = RcmpsCore.davidson(op,starts;nev,precondition=apply,options,callback)
    solve_seconds = time()-tick
    coefficients = copy(result.vectors)
    energies,residuals,refined_converged = copy(result.energies),copy(result.residuals),nothing
    tick = time()
    reference = refined_ctx = nothing
    corrections = NamedTuple[]
    if validate || refine_steps > 0
        refined_params = something(refined,refined_integration(ctx.state.integrator_param))
        refined_nodes = something(refined_kinetic_nodes,isnothing(op.kinetic_nodes) ? Some(nothing) :
                                  2op.kinetic_nodes)
        refined_ctx = ExcitationContext(ctx;integrator=refined_params)
        reference = with_context(op,refined_ctx;kinetic_nodes=refined_nodes)
    end
    if refine_steps > 0
        # Preconditioned corrections under the refined action, accepted only when the fresh
        # residual decreases (the D≤48 research refinement). The refined vector's own
        # refined Rayleigh quotient and residual replace the Davidson values.
        x = coefficients[1];level = fresh_level(reference,x)
        push!(corrections,(step=0,energy=level.energy,residual=level.residual,accepted=true))
        for step in 1:refine_steps
            level.residual <= options.tol && break
            r = level.action-level.energy*x
            d = apply(r);d -= x*dot(x,d);d /= norm(d)
            shifted = op(d)-level.energy*d
            alpha = dot(shifted,r)/real(dot(shifted,shifted))
            trial = x-alpha*d;trial /= norm(trial)
            candidate = fresh_level(reference,trial)
            accepted = candidate.residual < level.residual
            push!(corrections,(step=step,energy=candidate.energy,residual=candidate.residual,accepted=accepted))
            accepted || break
            x,level = trial,candidate
        end
        coefficients[1] = x
        energies[1],residuals[1],refined_converged = level.energy,level.residual,level.residual <= options.tol
    end
    validation = nothing
    if validate
        validation = map(coefficients) do x
            standard,fine = fresh_level(op,x),fresh_level(reference,x)
            bilinear = bilinear_level(ctx,op,x;kinetic_nodes=op.kinetic_nodes)
            refined_bilinear = bilinear_level(refined_ctx,reference,x;kinetic_nodes=reference.kinetic_nodes)
            values = (standard.energy,fine.energy,bilinear.energy,refined_bilinear.energy)
            imaginary = maximum(abs,(standard.imaginary,fine.imaginary,bilinear.imaginary,refined_bilinear.imaginary))
            spread = maximum(values)-minimum(values)
            residual = max(standard.residual,fine.residual)
            (standard_energy=standard.energy,standard_residual=standard.residual,
             refined_energy=fine.energy,refined_residual=fine.residual,
             bilinear_energy=bilinear.energy,refined_bilinear_energy=refined_bilinear.energy,
             energy_spread=spread,max_imaginary=imaginary,
             residual_target_met=residual <= validation_tol,
             consistent=spread <= validation_tol && imaginary <= validation_tol,
             accepted=spread <= validation_tol && imaginary <= validation_tol && residual <= acceptance_residual)
        end
    end
    validation_seconds = time()-tick
    # Validated levels are accepted on the research criteria (energy agreement, imaginary
    # parts, fresh residuals); without validation only a converged search (and refinement)
    # is accepted. Every requested level must be present.
    accepted = length(coefficients) == nev && (isnothing(validation) ?
        result.converged && something(refined_converged,true) : all(v->v.accepted,validation))
    vectors = [original_tangent(coordinates,tangent_matrix(coordinates,x)) for x in coefficients]
    return (energies=energies,vectors=vectors,coefficients=coefficients,
            residuals=residuals,converged=result.converged,stop_reason=result.stop_reason,
            refined_converged=refined_converged,accepted=accepted,nev=nev,
            validation_tol=validation_tol,acceptance_residual=acceptance_residual,
            numops=result.numops,action_calls=op.calls[],history=result.history,
            momentum=ctx.momentum,sector=coordinates.sector,parity=coordinates.parity,
            parity_error=coordinates.parity_error,gauge=coordinates.gauge,dimension=length(coordinates),
            metric_rank=coordinates.metric_rank,discarded_metric_weight=coordinates.discarded_metric_weight,
            metric_defect=coordinates.metric_defect,density_condition=coordinates.density_condition,
            start=startkind,davidson=options,
            preconditioner=isnothing(pre) ? nothing :
                (nodes=pre.nodes,skew=pre.skew,floor=pre.floor,minimum=minimum(pre.values)),
            corrections=corrections,validation=validation,
            validated_energies=isnothing(validation) ? nothing : [v.refined_bilinear_energy for v in validation],
            integration=ctx.state.integrator_param,
            refined_integration=isnothing(refined_ctx) ? nothing : refined_ctx.state.integrator_param,
            kinetic_nodes=op.kinetic_nodes,
            refined_kinetic_nodes=isnothing(reference) ? nothing : reference.kinetic_nodes,
            timings=(context=context_seconds,preconditioner=preconditioner_seconds,
                     solve=solve_seconds,validation=validation_seconds),
            model=model.name,coordinates=coordinates)
end

reported_energies(r) = isnothing(r.validated_energies) ? r.energies : r.validated_energies
# Energies of an accepted result, else a SolverFailure describing why it is not accepted.
function accepted_energies(r)
    r.accepted && return reported_energies(r)
    detail = if length(r.energies) < r.nev
        "only $(length(r.energies)) of $(r.nev) levels returned"
    elseif isnothing(r.validation)
        "residuals $(r.residuals), refined_converged=$(r.refined_converged)"
    else
        "levels $(findall(v->!v.accepted,r.validation)) rejected: max fresh residual " *
        "$(maximum(max(v.standard_residual,v.refined_residual) for v in r.validation)) " *
        "(limit $(r.acceptance_residual)), energy spread $(maximum(v.energy_spread for v in r.validation)) " *
        "and imaginary part $(maximum(v.max_imaginary for v in r.validation)) (limit $(r.validation_tol) each)"
    end
    throw(SolverFailure("iterative excitation levels at p=$(r.momentum) not accepted (stop: $(r.stop_reason), " *
        "$detail); use excitation_eigsolve or strict=false to inspect them"))
end

transfer_memory(D,p) = (iszero(p) ? 48 : 80)*D^4

"""
    excitation_eigsolve(s, p=0; model=ExcitationModel(s), nev=1, sector=:auto, kw...)

Lowest `nev` tangent-space excitation energies at momentum `p` by a preconditioned
Davidson iteration on Hamiltonian actions; no dense matrix is assembled. This is the
large-D route (the pipeline of the D=16–64 mass scans); `excitation_spectrum` is the
dense reference. Unlike the dense routines, it restricts to one parity sector by default
and uses a fixed kinetic grid; `validated_energies` also use refined integration.

Sector and coordinates (`TangentCoordinates`): `sector` (`:auto` → `:odd` for a Z₂
state, `:all` otherwise), `parity`, `parity_tol`, `gauge=:density`, `metric_tol=0.0`.

Operator: the model's actions with `kinetic_nodes=256` fixed kinetic θ nodes (the
kinetic part is then exactly linear; `nothing` refines adaptively per action), and
potential terms on `s.integrator_param` (window, `n_quad`, tolerances; linear to that
tolerance), `kinetic_backend`, `kinetic_options`, `method`, and `parallel=:both`
(terms and forward/backward trajectories run concurrently when Julia has threads).

Solver: `preconditioner=:kinetic` (coarse kinetic matrix on `preconditioner_nodes=8`
nodes), `:none` or a function; `start=:auto` (the field insertion φ(0)|Ψ⟩, else the
lowest coarse kinetic modes), `:field`, `:kinetic`, `:random`, coordinates or tangents
W in the gauge of `s`; `davidson` (default `DavidsonOptions(maxspace=max(8,2nev+2))`:
fresh restarts that recompute `nev` retained actions, residual target 1e-7, stall rule).
`callback(row, U, AU)` receives white sector coordinates (columns); convert them with
`original_tangent(r.coordinates, tangent_matrix(r.coordinates, U[:,k]))`.

Validation (`validate=true`): every returned vector gets fresh actions at the standard
and at `refined` settings (default: window +10, twice `n_quad`, tolerances /10, twice
the kinetic nodes), and two independent bilinear energies. `validated_energies` are the
refined bilinear ones. A level is `accepted` when its four energies agree and its
imaginary parts vanish within `validation_tol=1e-7`, and its fresh residuals are at most
`acceptance_residual=1e-5` (the research acceptance criteria). `refine_steps>0` (nev=1)
first applies preconditioned corrections under the refined action; the returned energy
and residual are then the refined vector's refined ones and `refined_converged` reports
whether that residual meets `tol` (`converged` and `stop_reason` describe the Davidson
run). Spreads and residuals are consistency measurements, not error bounds; the
background state is not tested.

Memory (`max_bond=64`): the dense transfer context retains ≈48D⁴ bytes at p=0 (≈80D⁴
otherwise) and peaks near twice that while it is built. Each concurrent kinetic action
allocates a 16D⁴-byte triangular factor per node (32D⁴ at p≠0; bilinear validation at
p≠0 forms three), and the preconditioner runs up to `Threads.nthreads()` such actions. Each concurrent potential action
stores ≈640·n_quad·D² bytes of trajectories.
Returned `vectors` are metric-normalized tangents W in the gauge of `s`.
"""
function excitation_eigsolve(s::State,p::Real=0.0;model=ExcitationModel(s),sector=:auto,parity=:auto,
        parity_tol=1e-12,gauge=:density,metric_tol=0.0,environment_tol=1e-10,max_bond=64,kw...)
    isfinite(p) || throw(ArgumentError("invalid momentum"))
    s.D <= max_bond || throw(ArgumentError("excitation_eigsolve is limited to D≤$max_bond (dense transfer factors retain ≈$(round(transfer_memory(s.D,p)/2^30;digits=1)) GiB); increase max_bond explicitly"))
    tick = time()
    coordinates = TangentCoordinates(s;sector,parity,parity_tol,gauge,metric_tol)
    check_levels_request(coordinates;kw...)
    ctx = ExcitationContext(coordinates.state,p;tol=environment_tol,max_bond,store_transfer=false)
    return solve_levels(coordinates,ctx,model;context_seconds=time()-tick,kw...)
end

# Shared coordinates and momentum-independent factors (a p=0 base, so only the current
# momentum's two shifted LU factors are alive); each momentum warm-starts from the
# previous eigenvectors unless `start` is supplied. With `strict`, stop at the first
# momentum whose levels are not accepted.
function iterative_dispersion(s::State,momenta;model=ExcitationModel(s),sector=:auto,parity=:auto,
        parity_tol=1e-12,gauge=:density,metric_tol=0.0,environment_tol=1e-10,max_bond=64,start=:auto,
        strict=true,kw...)
    s.D <= max_bond || throw(ArgumentError("excitation_eigsolve is limited to D≤$max_bond (dense transfer factors retain ≈$(round(transfer_memory(s.D,1)/2^30;digits=1)) GiB); increase max_bond explicitly"))
    tick = time()
    coordinates = TangentCoordinates(s;sector,parity,parity_tol,gauge,metric_tol)
    check_levels_request(coordinates;start,kw...)
    base = ExcitationContext(coordinates.state,0.0;tol=environment_tol,max_bond,store_transfer=false)
    seconds = time()-tick
    results = []
    for (k,p) in enumerate(momenta)
        tick = time()
        ctx = ExcitationContext(base;momentum=p)
        seed = k == 1 || start !== :auto ? start : last(results).coefficients
        push!(results,solve_levels(coordinates,ctx,model;start=seed,
                                   context_seconds=(k == 1 ? seconds : 0.0)+time()-tick,kw...))
        strict && accepted_energies(last(results))
    end
    return results
end
