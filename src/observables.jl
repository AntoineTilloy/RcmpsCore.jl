"""Mass-one smearing kernel. At x=0 it has an integrable inverse-square-root singularity."""
jsource(x::Real) = iszero(x) ? Inf : besselk(0.25,abs(x))/(2.0^2.25*sqrt(pi)*gamma(1.25)*abs(x)^0.25)

# x=t|t| removes the singularity exactly: 2|t|J(t²) -> 1/sqrt(pi).
transformed_source(t::Real) = iszero(t) ? inv(sqrt(pi)) : 2abs(t)*jsource(t*t)

struct Edge
    src::Int
    dst::Int
    left::Bool
    coefficient::ComplexF64
end
struct Observable
    channels::Int
    edges::Vector{Edge}
    commutator::Bool
    kinetic::Bool
end
vertex_spec(beta) = Observable(1,[Edge(1,1,true,beta),Edge(1,1,false,beta)],false,false)
function monomial_spec(n::Integer)
    n >= 0 || throw(ArgumentError("monomial degree must be nonnegative"))
    edges = Edge[]
    for k in 1:n
        push!(edges,Edge(k,k+1,true,k),Edge(k,k+1,false,k))
    end
    return Observable(n+1,edges,false,false)
end
kinetic_spec(commutator) = Observable(4,[Edge(1,2,true,sqrt(2)),Edge(1,3,false,sqrt(2)),
                                        Edge(3,4,true,sqrt(2)),Edge(2,4,false,sqrt(2))],commutator,true)

struct ODEWorkspace{Adjoint}
    state::State
    spec::Observable
    insertion::Matrix{ComplexF64}
    tmp::Matrix{ComplexF64}
end
function ode_rhs!(du,u,p::ODEWorkspace{adj},t) where adj
    s, spec = p.state,p.spec
    jac, j = 2abs(t),transformed_source(t)
    @views for k in 1:spec.channels
        if adj
            lindblad_adjoint!(du[:,:,k],s,u[:,:,k],p.tmp)
        else
            lindblad!(du[:,:,k],s,u[:,:,k],p.tmp)
        end
        du[:,:,k] .*= jac
    end
    @views for e in spec.edges
        a = e.coefficient*j
        if adj
            if e.left
                mul!(du[:,:,e.src],u[:,:,e.dst],p.insertion,a,1)
            else
                mul!(du[:,:,e.src],p.insertion',u[:,:,e.dst],a,1)
            end
        elseif e.left
            mul!(du[:,:,e.dst],p.insertion,u[:,:,e.src],a,1)
        else
            mul!(du[:,:,e.dst],u[:,:,e.src],p.insertion',a,1)
        end
    end
    return nothing
end

function quadrature(p::IntegratorParameters)
    p.n_quad >= 8 && p.max_point > 0 && p.atol > 0 && p.rtol > 0 && p.maxiters > 0 ||
        throw(ArgumentError("invalid integration controls"))
    n = p.n_quad
    k = collect(1:n-1)
    f = eigen(SymTridiagonal(zeros(n),k./sqrt.(4 .* k.^2 .- 1)))
    a = sqrt(p.max_point)
    points = a/2 .* (f.values .+ 1)
    weights = a .* abs2.(f.vectors[1,:])
    return vcat(-reverse(points),points),vcat(reverse(weights),weights)
end

function trajectory(s,spec,op,points,adj,method;value_only=false)
    p = s.integrator_param
    a = sqrt(p.max_point)
    u0 = zeros(ComplexF64,s.D,s.D,spec.channels)
    if adj
        u0[:,:,end] .= Matrix{ComplexF64}(I,s.D,s.D)
    else
        u0[:,:,1] .= s.ρ0
    end
    ws = ODEWorkspace{adj}(s,spec,op,similar(s.R))
    prob = ODEProblem(ode_rhs!,u0,(-a,a),ws)
    # Even sources allow a forward solve of the adjoint, indexed at -t.
    savepoints = value_only ? [a] : vcat(points,a)
    sol = solve(prob,method;saveat=savepoints,save_start=false,save_end=true,
                save_everystep=false,dense=false,tstops=[0.0],
                abstol=p.atol,reltol=p.rtol,maxiters=p.maxiters)
    successful_retcode(sol) || throw(SolverFailure("ODE solve failed: $(sol.retcode)"))
    all(isfinite,sol.u[end]) || throw(SolverFailure("nonfinite observable trajectory"))
    return sol
end

function addtriple!(out,a,b,c,coefficient,tmp)
    mul!(tmp,a,b)
    mul!(out,tmp,c,coefficient,1)
    return out
end

function check_parallel(parallel)
    parallel in (:serial,:terms,:trajectories,:both) ||
        throw(ArgumentError("parallel must be :serial, :terms, :trajectories, or :both"))
end

# Join every task before returning or throwing. Preserve SolverFailure so that
# line searches can reject a failed trial, rather than seeing TaskFailedException.
function evaluation_map(f,items,threaded)
    (!threaded || Threads.nthreads() == 1) && return map(f,items)
    tasks = map(items) do item
        Threads.@spawn try
            (f(item),nothing)
        catch error
            (nothing,error)
        end
    end
    results = map(fetch,tasks)
    for (_,error) in results
        isnothing(error) || throw(error)
    end
    return map(first,results)
end

"""Evaluate an observable and its Riemannian gradient of the REAL PART.
The returned complex value is retained for vertex operators. For any horizontal W,
`d real(value)[W] = metric(s, gradient, W)`. Set `boundary=false` only for research.
"""
function evaluate(s::State,spec::Observable;gradient=true,boundary=true,method=Vern7(),
                  boundary_method=:iterative, raw=false, grid=nothing,parallel=:serial,
                  kinetic_backend=:ode,kinetic_options=KineticOptions(),
                  krylov_options=KineticKrylovOptions())
    check_parallel(parallel)
    kinetic_backend in (:ode,:resolvent,:krylov) || throw(ArgumentError("unknown kinetic backend"))
    if spec.kinetic && kinetic_backend!=:ode
        result = kinetic_resolvent(s,[(1.,spec)];gradient,options=kinetic_options,boundary_method,
                                   backend=kinetic_backend,krylov_options)
        gradient || return result
        value,covector,a = result
        return raw ? result : (value,finish_gradient(s,covector,a;boundary,boundary_method))
    end
    points,weights = gradient ? (isnothing(grid) ? quadrature(s.integrator_param) : grid) : (Float64[],Float64[])
    gradient || quadrature_check(s.integrator_param)
    op = spec.commutator ? s.Q*s.R-s.R*s.Q : s.R
    if gradient
        forward,backward = evaluation_map((false,true),parallel in (:trajectories,:both)) do adj
            trajectory(s,spec,op,points,adj,method)
        end
    else
        forward = trajectory(s,spec,op,points,false,method;value_only=true)
    end
    value = tr(@view forward.u[end][:,:,end])
    gradient || return value
    covector,a = observable_pullback(s,spec,points,weights,forward,backward)
    raw && return value,covector,a
    return value,finish_gradient(s,covector,a;boundary,boundary_method)
end

function observable_pullback(s,spec,points,weights,forward,backward)
    m,n,u,v,tmp = (zero(s.R) for _ in 1:5)
    @views for i in eachindex(points)
        rho,obs = forward.u[i],backward.u[length(points)-i+1]
        w = weights[i]*2abs(points[i])
        for k in 1:spec.channels
            x,a = rho[:,:,k],obs[:,:,k]
            addtriple!(m,x,s.R',a,w,tmp)
            addtriple!(m,x,a,s.R',-w,tmp)
            addtriple!(n,a,s.R,x,w,tmp)
            addtriple!(n,s.R,a,x,-w,tmp)
        end
        j = weights[i]*transformed_source(points[i])
        for e in spec.edges
            if e.left
                mul!(u,rho[:,:,e.src],obs[:,:,e.dst],j*e.coefficient,1)
            else
                mul!(v,obs[:,:,e.dst],rho[:,:,e.src],j*e.coefficient,1)
            end
        end
    end
    if spec.commutator
        # Pull back C=[Q,R], with delta Q=-R'W on the horizontal tangent.
        mul!(m,u,s.Q,1,1); mul!(m,s.Q,u,-1,1)
        addtriple!(m,s.R,u,s.R',-1,tmp)
        addtriple!(m,u,s.R,s.R',1,tmp)
        mul!(n,s.Q',v,1,1); mul!(n,v,s.Q',-1,1)
        addtriple!(n,s.R,v,s.R',-1,tmp)
        addtriple!(n,s.R,s.R',v,1,tmp)
    else
        m .+= u
        n .+= v
    end
    # For complex beta, both Wirtinger terms are needed for real(value).
    covector = (m+n')/2
    a = backward.u[end][:,:,1]
    a = (a+a')/2
    return covector,a
end

function finish_gradient(s,covector,a;boundary=true,boundary_method=:iterative)
    if boundary
        tmp = similar(s.R)
        h = boundary_environment(s,a;method=boundary_method)
        addtriple!(covector,s.ρ0,s.R',h,1,tmp)
        addtriple!(covector,s.ρ0,h,s.R',-1,tmp)
    end
    reg = s.manifold_param.metric_reg
    reg >= 0 || throw(ArgumentError("metric regularizer must be nonnegative"))
    grad = Matrix((Hermitian(s.ρ0+reg*I)\covector)')
    return grad
end
function quadrature_check(p)
    p.max_point > 0 && p.atol > 0 && p.rtol > 0 && p.maxiters > 0 ||
        throw(ArgumentError("invalid integration controls"))
end
vertex(s::State,beta::Number;kw...) = evaluate(s,vertex_spec(beta);kw...)
function real_observable(s,spec;gradient=true,kw...)
    result = evaluate(s,spec;gradient,kw...)
    return gradient ? (real(result[1]),result[2]) : real(result)
end
phi_monomial(s::State,n::Integer;kw...) = real_observable(s,monomial_spec(n);kw...)
kinetic1(s::State;kw...) = real_observable(s,kinetic_spec(false);kw...)
kinetic2(s::State;kw...) = real_observable(s,kinetic_spec(true);kw...)

function linear_combination(s,terms;gradient=true,boundary=true,boundary_method=:iterative,
                            parallel=:serial,kinetic_backend=:ode,
                            kinetic_options=KineticOptions(),krylov_options=KineticKrylovOptions(),kw...)
    check_parallel(parallel)
    kinetic_backend in (:ode,:resolvent,:krylov) || throw(ArgumentError("unknown kinetic backend"))
    value = 0.0
    covector,a = zero(s.R),zero(s.R)
    grid = gradient ? quadrature(s.integrator_param) : nothing
    active = filter(term->!iszero(first(term)),terms)
    kinetics = kinetic_backend!=:ode ? filter(term->term[2].kinetic,active) : []
    jobs = isempty(kinetics) ? active : vcat([(1.,nothing)],filter(term->!term[2].kinetic,active))
    results = evaluation_map(jobs,parallel in (:terms,:both)) do (_,spec)
        isnothing(spec) ? kinetic_resolvent(s,kinetics;gradient,options=kinetic_options,boundary_method,
                                           backend=kinetic_backend,krylov_options) :
            evaluate(s,spec;gradient,raw=true,grid,parallel,kw...)
    end
    # Reduction order is fixed, independent of task completion order.
    for ((coefficient,_),result) in zip(jobs,results)
        if gradient
            v,c,b = result
            value += real(coefficient*v)
            covector .+= coefficient .* c
            a .+= coefficient .* b
        else
            value += real(coefficient*result)
        end
    end
    return gradient ? (value,finish_gradient(s,covector,a;boundary,boundary_method)) : value
end
phi4_terms(g::Real) = [(1.0,kinetic_spec(false)),(1.0,kinetic_spec(true)),
                       (Float64(g),monomial_spec(4))]
fg_phi4(s::State,g::Real;kw...) = linear_combination(s,phi4_terms(g);kw...)
function sgordon_terms(beta::Number)
    iszero(beta) && throw(ArgumentError("beta=0 has a divergent additive constant"))
    isreal(beta^2) || throw(ArgumentError("beta must be real or purely imaginary"))
    c = real(1/(2beta^2))
    return [(1.0,kinetic_spec(false)),(1.0,kinetic_spec(true)),
            (c,vertex_spec(beta)),(c,vertex_spec(-beta)),(-0.5,monomial_spec(2))]
end
fg_sgordon(s::State,beta::Number;kw...) = linear_combination(s,sgordon_terms(beta);kw...)
Y(m1,m2) = (m2^2-m1^2+2m1^2*log(m1/m2))/(8pi)
Z(m1,m2) = log(m2/m1)/(2pi)
function phi4_massmod_terms(g::Real,massmod)
    massmod > 0 || throw(ArgumentError("massmod must be positive"))
    z = Z(massmod,1)
    c2 = (massmod^2-1)/2 - 6g*massmod^2*z
    return [(1/massmod^2,kinetic_spec(false)),(1/massmod^2,kinetic_spec(true)),
             (g,monomial_spec(4)),(c2/massmod^2,monomial_spec(2))]
end
function fg_phi4_massmod(s::State,g::Real,massmod=0.1;gradient=true,kw...)
    terms = phi4_massmod_terms(g,massmod)
    z = Z(massmod,1)
    result = linear_combination(s,terms;gradient,kw...)
    constant = Y(massmod,1)/massmod^2+3g*z^2
    return gradient ? (result[1]+constant,result[2]) : result+constant
end
