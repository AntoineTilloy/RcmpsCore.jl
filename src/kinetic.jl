"""Accuracy controls for the infinite-line kinetic resolvent backend.
Node doubling checks energy and the full stationary-response gradient. This is
an observed convergence test, not a certified error bound. Dense Schur costs O(D⁶).
An opt-in `gradient_fallback_atol` permits a plateau below that absolute error
in the state's regularized metric. Two consecutive refinements must satisfy
the energy check and metric ceiling; the latest gradient difference must be
at least `gradient_plateau_ratio` times the preceding one. Zero disables fallback.
"""
Base.@kwdef struct KineticOptions
    minnodes::Int = 32
    maxnodes::Int = 512
    atol::Float64 = 1e-11
    rtol::Float64 = 1e-10
    gradient_fallback_atol::Float64 = 0.0
    gradient_plateau_ratio::Float64 = 0.5
end

function kinetic_grid(n)
    k = collect(1:n-1)
    f = eigen(SymTridiagonal(zeros(n),k./sqrt.(4 .* k.^2 .- 1)))
    return pi/4 .* (f.values .+ 1),abs2.(f.vectors[1,:]) # includes the factor 2/pi
end

# Exact kernel convolution gives (2/pi)∫₀^{π/2} tr[C(I-cos(θ)L)⁻¹(ρC†)]dθ.
# Both kinetic terms reuse this Schur factorization. The singular source has
# been integrated out analytically; this computes the infinite-line observable.
function kinetic_components(s,spec,ctx,grid;gradient=true)
    op = spec.commutator ? s.Q*s.R-s.R*s.Q : s.R
    b,c = ctx.Z'*vec(s.ρ0*op'),ctx.Z'*vec(op')
    m,n,u,v,env = (zero(s.R) for _ in 1:5)
    value = 0.0
    for (theta,w) in zip(grid...)
        z = cos(theta)
        factor = UpperTriangular(I-z*ctx.T)
        xx = factor\b
        value += w*real(dot(c,xx))
        gradient || continue
        x = reshape(ctx.Z*xx,s.D,s.D)
        a = reshape(ctx.Z*(factor'\c),s.D,s.D)'
        m .+= w*z .* (x*s.R'*a-x*a*s.R')
        n .+= w*z .* (a*s.R*x-s.R*a*x)
        u .+= w .* x
        v .+= w .* (a*s.ρ0)
        env .+= w .* (op'*a)
    end
    gradient || return value
    if spec.commutator
        m .+= u*s.Q-s.Q*u-s.R*u*s.R'+u*s.R*s.R'
        n .+= s.Q'*v-v*s.Q'-s.R*v*s.R'+s.R*s.R'*v
    else
        m .+= u
        n .+= v
    end
    return value,(m+n')/2,(env+env')/2
end

function kinetic_sum(s,terms,ctx,n;gradient)
    grid = kinetic_grid(n)
    e,c,a = 0.0,zero(s.R),zero(s.R)
    for (weight,spec) in terms
        result = kinetic_components(s,spec,ctx,grid;gradient)
        if gradient
            value,covector,env = result
            e += weight*value; c .+= weight .* covector; a .+= weight .* env
        else
            e += weight*result
        end
    end
    return gradient ? (e,c,a) : e
end

kinetic_context(s,::Val{:resolvent},options) = schur(generate_T(s))

function kinetic_resolvent(s,terms;gradient=true,options=KineticOptions(),
                           boundary_method=:iterative,backend=:resolvent,
                           krylov_options=KineticKrylovOptions())
    p = options
    p.minnodes>=2 && p.maxnodes>=2p.minnodes && isfinite(p.atol) && isfinite(p.rtol) &&
        p.atol>0 && p.rtol>=0 &&
        isfinite(p.gradient_fallback_atol) && p.gradient_fallback_atol>=0 &&
        isfinite(p.gradient_plateau_ratio) && 0<p.gradient_plateau_ratio<=1 ||
        throw(ArgumentError("invalid kinetic quadrature controls"))
    ctx = kinetic_context(s,Val(backend),krylov_options)
    n = p.minnodes
    previous = kinetic_sum(s,terms,ctx,n;gradient)
    previous_g = gradient ? finish_gradient(s,copy(previous[2]),previous[3];boundary_method) : nothing
    previous_metric_difference = Inf
    previous_energy_pass = false
    while 2n<=p.maxnodes
        n *= 2
        current = kinetic_sum(s,terms,ctx,n;gradient)
        if gradient
            e,c,a = current
            g = finish_gradient(s,copy(c),a;boundary_method)
            difference = g-previous_g
            de,dg,gn = abs(e-previous[1]),norm(difference),norm(g)
            isfinite(e) && isfinite(gn) && isfinite(de) && isfinite(dg) ||
                throw(SolverFailure("nonfinite kinetic quadrature convergence check"))
            energy_pass = de<=p.atol+p.rtol*abs(e)
            if energy_pass && dg<=p.atol+p.rtol*gn
                return current
            end
            if p.gradient_fallback_atol>0
                dm = sqrt(metric(s,difference,difference))
                isfinite(dm) || throw(SolverFailure("nonfinite kinetic metric gradient difference"))
                if energy_pass && previous_energy_pass &&
                    max(dm,previous_metric_difference)<=p.gradient_fallback_atol &&
                    dm>=p.gradient_plateau_ratio*previous_metric_difference
                    @warn "Kinetic quadrature accepted gradient plateau" nodes=n metric_difference=dm previous_metric_difference ceiling=p.gradient_fallback_atol energy_difference=de
                    return current
                end
                previous_metric_difference = dm
                previous_energy_pass = energy_pass
            end
            previous_g = g
        else
            isfinite(current) && isfinite(previous) ||
                throw(SolverFailure("nonfinite kinetic quadrature convergence check"))
            abs(current-previous)<=p.atol+p.rtol*abs(current) && return current
        end
        previous = current
    end
    throw(SolverFailure("kinetic resolvent quadrature did not converge by $n nodes"))
end
