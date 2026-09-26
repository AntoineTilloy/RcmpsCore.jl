"""Controls for `davidson`, a restarted Hermitian Rayleigh–Ritz iteration.

`maxspace` bounds the search space. At a restart the `retain` lowest Ritz vectors
are kept (`nothing` means `nev`). With `refresh=true` their actions are recomputed
rather than recombined from stored columns: this "fresh restart" keeps the projected
matrix consistent with the operator when actions carry integration error. With
`refresh=false` the restart is a thick restart using stored action columns.

A search stops when every targeted residual is at most `tol` (`:converged`), after
`maxops` operator calls (`:maxops`), when it stalls (`:stalled`), or when no new
direction orthogonal to the search space remains while a residual exceeds `tol`
(`:breakdown`, e.g. a spanned sector whose residual is set by the anti-Hermitian part
of an inexact operator). It stalls when, over steps that carry every targeted root
(at least `stall_minops` of them and more than `stall_window`), the largest targeted
residual is at most `stall_residual_limit` (default `max(1e-5, 10tol)`), the best
residual improved by less than a fraction `stall_rtol` over the last `stall_window`
such steps, and every targeted Ritz energy varied by at most `stall_energy_atol`.
`stall_window=0` disables it. Only `:converged` is reported as converged.
"""
Base.@kwdef struct DavidsonOptions
    tol::Float64 = 1e-7
    maxops::Int = 200
    maxspace::Int = 8
    retain::Union{Nothing,Int} = nothing
    refresh::Bool = true
    hermitian_tol::Float64 = 1e-7
    stall_window::Int = 20
    stall_minops::Int = 40
    stall_rtol::Float64 = 0.1
    stall_residual_limit::Union{Nothing,Float64} = nothing
    stall_energy_atol::Float64 = 1e-9
end

stall_limit(o::DavidsonOptions) = something(o.stall_residual_limit,max(1e-5,10o.tol))

function check_davidson(o::DavidsonOptions,nev)
    retain = something(o.retain,nev)
    nev >= 1 || throw(ArgumentError("nev must be positive"))
    isfinite(o.tol) && o.tol > 0 && o.maxops >= 1 && isfinite(o.hermitian_tol) && o.hermitian_tol > 0 ||
        throw(ArgumentError("invalid Davidson controls"))
    nev <= retain < o.maxspace || throw(ArgumentError("require nev <= retain < maxspace (nev=$nev, retain=$retain, maxspace=$(o.maxspace))"))
    o.maxops >= nev || throw(ArgumentError("maxops must be at least nev"))
    o.stall_window >= 0 || throw(ArgumentError("stall_window must be nonnegative"))
    if o.stall_window > 0
        limit = stall_limit(o)
        o.stall_minops >= 1 && isfinite(o.stall_rtol) && 0 < o.stall_rtol < 1 &&
            isfinite(o.stall_energy_atol) && o.stall_energy_atol >= 0 ||
            throw(ArgumentError("invalid Davidson stagnation controls"))
        isfinite(limit) && limit >= o.tol ||
            throw(ArgumentError("stall_residual_limit ($limit) must be at least tol ($(o.tol))"))
    end
    return retain
end

# Rows need `residual` (largest targeted residual) and `energies`. Only rows that carry
# every targeted root enter: a partially rebuilt space after a fresh restart does not.
function davidson_stalled(history,o::DavidsonOptions)
    window = o.stall_window
    window > 0 && !isempty(history) || return false
    nev = length(history[end].energies)
    full = [row for row in history if length(row.energies) == nev]
    n = length(full)
    n >= max(o.stall_minops,window+1) || return false
    full[end].residual <= stall_limit(o) || return false
    old_best = minimum(row.residual for row in @view full[1:n-window])
    best = minimum(row.residual for row in full)
    best >= (1-o.stall_rtol)*old_best || return false
    recent = @view full[n-window:n]
    for k in 1:nev
        values = [row.energies[k] for row in recent]
        maximum(values)-minimum(values) <= o.stall_energy_atol || return false
    end
    return true
end

# Two modified Gram–Schmidt passes avoid losing a new direction to roundoff.
function orthogonalize!(x,basis)
    for _ in 1:2, b in basis
        x .-= b .* dot(b,x)
    end
    return x
end

"""
    davidson(op, x0; nev=1, precondition=identity, options=DavidsonOptions(), callback=nothing)

Lowest `nev` eigenpairs of a Hermitian linear map known only through `op(x)`.
`x0` is a start vector or a collection of start vectors. `precondition(r)` maps a
residual to a correction direction (a fixed positive approximation of the inverse
operator is appropriate); the expansion vector is `-precondition(r)`.

The projected matrix `V'AV` is formed from the actual action columns. Its
anti-Hermitian part must stay below `options.hermitian_tol`, otherwise a
`SolverFailure` is thrown. Residuals are recomputed from stored actions at every
step, so they are exact for the operator as evaluated, not recursive estimates.
`callback(row, U, AU)` receives the history row (count, cycle, dimension, energies,
residuals, energy, residual, hermiticity, orthogonality, refresh_change) and n×m
matrices whose columns are the normalized current Ritz vectors and their actions.
At most `maxspace-1` independent start vectors are accepted.

Returns energies, normalized vectors, their actions, residuals, `converged`,
`stop_reason` (`:converged`, `:stalled`, `:maxops` or `:breakdown`), `numops` and
`history`. A `:maxops` stop always returns `nev` pairs: the budget is never ended in
the middle of rebuilding a freshly restarted space.
"""
function davidson(op,x0;nev::Integer=1,precondition=identity,options::DavidsonOptions=DavidsonOptions(),
                  callback=nothing)
    o = options
    retain = check_davidson(o,nev)
    starts = x0 isa AbstractVector{<:Number} ? [x0] : collect(x0)
    isempty(starts) && throw(ArgumentError("at least one start vector is required"))
    pending = Vector{ComplexF64}[]
    for x in starts
        y = Vector{ComplexF64}(x)
        all(isfinite,y) || throw(ArgumentError("start vectors must be finite"))
        orthogonalize!(y,pending)
        scale = norm(x)
        scale > 0 && norm(y) > sqrt(eps())*scale && push!(pending,y/norm(y))
    end
    isempty(pending) && throw(ArgumentError("start vectors must be nonzero and independent"))
    length(pending) < o.maxspace ||
        throw(ArgumentError("at most maxspace-1=$(o.maxspace-1) start vectors are accepted"))
    n = length(first(pending))
    all(x->length(x)==n,pending) || throw(DimensionMismatch("start vectors differ in length"))
    nev <= n || throw(ArgumentError("nev=$nev exceeds the dimension $n"))
    stale = Union{Nothing,Vector{ComplexF64}}[nothing for _ in pending]
    vs,avs = Vector{ComplexF64}[],Vector{ComplexF64}[]
    history = NamedTuple[]
    cycle = 1
    for count in 1:o.maxops
        x = popfirst!(pending);previous = popfirst!(stale)
        orthogonalize!(x,vs);x ./= norm(x)
        ax = Vector{ComplexF64}(op(x))
        length(ax) == n && all(isfinite,ax) || throw(SolverFailure("invalid operator action"))
        change = isnothing(previous) ? 0.0 : norm(ax-previous)
        push!(vs,x);push!(avs,ax)
        v,av = reduce(hcat,vs),reduce(hcat,avs)
        projected = v'*av
        defect = norm(projected-projected')/max(1,norm(projected))
        defect <= o.hermitian_tol ||
            throw(SolverFailure("Davidson projected operator is not Hermitian: $defect"))
        f = eigen(Hermitian((projected+projected')/2))
        m = min(nev,length(vs))
        u,au = v*f.vectors[:,1:m],av*f.vectors[:,1:m]
        for k in 1:m
            scale = norm(@view u[:,k])
            u[:,k] ./= scale;au[:,k] ./= scale
        end
        energies = [real(dot(u[:,k],au[:,k])) for k in 1:m]
        r = au-u*Diagonal(energies)
        residuals = [norm(@view r[:,k]) for k in 1:m]
        row = (count=count,cycle=cycle,dimension=length(vs),energies=energies,residuals=residuals,
               energy=energies[1],residual=maximum(residuals),hermiticity=defect,
               orthogonality=norm(v'*v-I),refresh_change=change)
        push!(history,row)
        isnothing(callback) || callback(row,u,au)
        converged = m == nev && row.residual <= o.tol
        stalled = !converged && m == nev && davidson_stalled(history,o)
        finish(reason) = (energies=energies,vectors=[u[:,k] for k in 1:m],actions=[au[:,k] for k in 1:m],
                          residuals=residuals,converged=reason == :converged,stop_reason=reason,
                          numops=count,history=history)
        converged && return finish(:converged)
        stalled && return finish(:stalled)
        count == o.maxops && return finish(:maxops)
        isempty(pending) || continue # still rebuilding the retained space after a fresh restart
        if length(vs) >= o.maxspace
            keep = f.vectors[:,1:min(retain,length(vs))]
            kept,keptactions = v*keep,av*keep
            cycle += 1
            # A fresh rebuild needs nev calls before all targeted roots are present again.
            o.refresh && o.maxops-count < nev && return finish(:maxops)
            if o.refresh
                empty!(vs);empty!(avs)
                pending = [kept[:,k] for k in axes(kept,2)]
                stale = Union{Nothing,Vector{ComplexF64}}[keptactions[:,k] for k in axes(kept,2)]
                continue
            end
            vs = [kept[:,k] for k in axes(kept,2)];avs = [keptactions[:,k] for k in axes(kept,2)]
        end
        # Expand along the lowest unconverged root whose residual leaves the search space.
        function expansion(k)
            y = -Vector{ComplexF64}(precondition(r[:,k]))
            length(y) == n && all(isfinite,y) || throw(SolverFailure("invalid preconditioned residual"))
            orthogonalize!(y,vs)
            if norm(y) < 1e-13*max(1,residuals[k])
                y = r[:,k];orthogonalize!(y,vs)
            end
            return norm(y) > eps()*max(1,residuals[k]) ? y : nothing
        end
        candidates = findall(>(o.tol),residuals)
        isempty(candidates) && (candidates = [m])
        y = nothing
        for k in candidates
            y = expansion(k)
            isnothing(y) || break
        end
        if isnothing(y)
            # Every targeted residual lies in the search space: with every root present no
            # direction is left (e.g. a spanned sector at the anti-Hermitian floor).
            m == nev && return finish(:breakdown)
            # An invariant subspace smaller than nev: continue with a deterministic direction.
            y = randn(MersenneTwister(count),ComplexF64,n);orthogonalize!(y,vs)
        end
        push!(pending,y/norm(y));push!(stale,nothing)
    end
end
