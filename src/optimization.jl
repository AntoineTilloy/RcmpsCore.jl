metric(s::State,w1,w2) = 2real(dot(w1,w2*(s.ρ0+s.manifold_param.metric_reg*I)))
function precondition(s::State,w)
    p = s.manifold_param
    p.precond_reg > 0 || throw(ArgumentError("preconditioner regularizer must be positive"))
    return Matrix((Hermitian(s.ρ0+p.precond_reg*I)\(w*(s.ρ0+p.metric_reg*I))')')
end
function retraction(s::State,w::AbstractMatrix,alpha::Real)
    size(w) == size(s.R) || throw(DimensionMismatch("tangent has the wrong dimensions"))
    k = s.K + (im*alpha/2)*(w'*s.R-s.R'*w)
    out = State(s.R+alpha*w,k)
    copy_param!(out,s)
    return out,w
end
# Retractions must have derivative w at zero. The old nonlinear step cap broke
# the line search's interpretation of alpha. Use the canonical curve for both names.
retraction_regulated(s::State,w,alpha;kw...) = retraction(s,w,alpha)
naive_transport!(w,sold,d,alpha,snew) = w
function finalize!(s,f,g,numiter)
    p = s.manifold_param
    p.precond_reg = max(10eps(),p.precond_reg_prefactor*min(metric(s,g,g),1.0))
    return s,f,g
end

"""Bounded Armijo backtracking; a failed search never returns an uphill step."""
Base.@kwdef struct BackwardLineSearch <: OptimKit.AbstractLineSearch
    c1::Float64 = 1e-4
    shrink::Float64 = 0.5
    maxiter::Int = 30
end
function (ls::BackwardLineSearch)(fg,x,p,fg0=fg(x);retract=retraction,inner=metric,
                                initialguess=1.0,acceptfirst=false,kw...)
    f0,g0 = fg0
    slope = inner(x,g0,p)
    slope < 0 || throw(SolverFailure("line search direction is not a descent direction"))
    alpha = initialguess
    failed_trials=0
    last_failure=""
    for k in 1:ls.maxiter
        try
            y,v = retract(x,p,alpha)
            f,g = fg(y)
            if isfinite(f) && f <= f0+ls.c1*alpha*slope
                return y,f,g,v,alpha,k
            end
        catch e
            e isa SolverFailure || rethrow()
            failed_trials+=1
            last_failure=sprint(showerror,e)
            # Failed numerical trial points are rejected; the accepted state is retained.
        end
        alpha *= ls.shrink
    end
    detail=failed_trials>0 ? "; $failed_trials numerical trial failures; last cause: $last_failure" : ""
    throw(SolverFailure("Armijo line search exhausted $(ls.maxiter) steps$detail"))
end
"""Wolfe search with bounded parameter excursions and checked Armijo recovery.
Approximate Wolfe steps may change energy within `energy_atol`, representing
numerical evaluation error. Their slope conditions are still checked explicitly.
The step bound leaves the canonical curve intact.
"""
struct SafeguardedLineSearch{L} <: OptimKit.AbstractLineSearch
    wolfe::L
    max_relative_step::Float64
    backtracking::BackwardLineSearch
    energy_atol::Float64
end
function SafeguardedLineSearch(;max_relative_step=0.5,energy_atol=1e-10)
    isfinite(energy_atol) && energy_atol>=0 || throw(ArgumentError("invalid energy tolerance"))
    return SafeguardedLineSearch(OptimKit.HagerZhangLineSearch(ϵ=energy_atol),
        Float64(max_relative_step),BackwardLineSearch(),Float64(energy_atol))
end

function (ls::SafeguardedLineSearch)(fg,x,p,fg0=fg(x);retract=retraction,inner=metric,
                                   initialguess=1.0,acceptfirst=false,kw...)
    ls.max_relative_step > 0 || throw(ArgumentError("step bound must be positive"))
    slope0=inner(x,fg0[2],p)
    isfinite(slope0) && slope0<0 || throw(SolverFailure("line search direction is not a descent direction"))
    alpha_max = ls.max_relative_step*max(1,norm(x.R))/max(norm(p),eps())
    calls = Ref(0)
    counted(y) = (calls[] += 1; fg(y))
    bounded(x,p,alpha) = begin
        abs(alpha) <= alpha_max || throw(SolverFailure("line search trial exceeds step bound"))
        retract(x,p,alpha)
    end
    try
        y,f,g,v,alpha,_ = ls.wolfe(counted,x,p,fg0;retract=bounded,inner,
                                 initialguess=min(initialguess,alpha_max),acceptfirst,kw...)
        # OptimKit can also return its last trial after exhausting its search.
        # Recheck its conditions instead of accepting such a trial unconditionally.
        slope=inner(y,g,v)
        c1,c2=ls.wolfe.c₁,ls.wolfe.c₂
        exact=(f-fg0[1] <= c1*alpha*slope0) && (slope >= c2*slope0)
        approximate=(f <= fg0[1]+ls.energy_atol) && (c2*slope0 <= slope <= (2c1-1)*slope0)
        isfinite(f) && isfinite(slope) && alpha>0 && (exact || approximate) ||
            throw(SolverFailure("Wolfe search returned a trial without valid Wolfe conditions"))
        return y,f,g,v,alpha,calls[]
    catch e
        e isa SolverFailure || rethrow()
        y,f,g,v,alpha,_ = ls.backtracking(counted,x,p,fg0;retract=bounded,inner,
                                        initialguess=min(initialguess,alpha_max))
        return y,f,g,v,alpha,calls[]
    end
end
const ls_backward = BackwardLineSearch()
const ls_default = SafeguardedLineSearch()
const ls_lenient = OptimKit.HagerZhangLineSearch(c₁=1e-2,c₂=0.99,ϵ=1e-5)

function energy_function(s,massmod;parallel=:serial,kinetic_backend=:ode,
                         kinetic_options=KineticOptions(),krylov_options=KineticKrylovOptions())
    name,c = s.model_param.name,s.model_param.coupling
    isreal(c) || throw(ArgumentError("model coupling must be real; sg supplies its own factor i"))
    options = (;parallel,kinetic_backend,kinetic_options,krylov_options)
    name == "phi4" && return x->fg_phi4(x,real(c);options...)
    name == "phi4_m" && return x->fg_phi4_massmod(x,real(c),massmod;options...)
    name == "shg" && return x->fg_sgordon(x,real(c);options...)
    name == "sg" && return x->fg_sgordon(x,im*real(c);options...)
    throw(ArgumentError("unknown model: $name"))
end

"""Optimize using released OptimKit (GD, CG, or LBFGS). Set `diagnostics=true`
to return energy, gradient norm, convergence, evaluation count, and history too.
"""
function optimize_state(s::State;method="LBFGS",n_iter=1000,gradtol=1e-7,
                        retract=retraction,transport! = naive_transport!,
                        precond=precondition,inner=metric,mlbfgs=20,verbosity=2,
                        ls=ls_default,massmod=1.0,diagnostics=false,parallel=:serial,
                        kinetic_backend=:ode,kinetic_options=KineticOptions(),
                        krylov_options=KineticKrylovOptions())
    n_iter >= 0 && gradtol > 0 || throw(ArgumentError("invalid optimization controls"))
    check_parallel(parallel)
    fg = energy_function(s,massmod;parallel,kinetic_backend,kinetic_options,krylov_options)
    options = (;maxiter=n_iter,gradtol,verbosity,linesearch=ls)
    alg = method == "LBFGS" ? OptimKit.LBFGS(mlbfgs;options...) :
          method == "GD" ? OptimKit.GradientDescent(;options...) :
          method == "CG" ? OptimKit.ConjugateGradient(;options...) :
          throw(ArgumentError("unknown optimization method: $method"))
    out,f,g,numfg,history = OptimKit.optimize(fg,deepcopy(s),alg;inner,retract,transport!,
                       precondition=precond,finalize!,isometrictransport=false)
    out.e0,out.grad = f,Matrix(g)
    gn = sqrt(inner(out,g,g))
    return diagnostics ? (state=out,energy=f,gradient_norm=gn,converged=gn<=gradtol,
                          evaluations=numfg,history=history) : out
end
