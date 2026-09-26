"""Hamiltonian assembled from the same normal-ordered terms as the ground-state
objectives. Supported names: :phi4, :shg, :sg, :phi4_m. Bare mass and c are one;
:phi4_m uses exactly the rescaled energy units of fg_phi4_massmod.
Additive constants cancel from excitation energies.
"""
struct ExcitationModel
    name::Symbol
    terms::Vector{Tuple{Float64,Observable}}
end
function ExcitationModel(name::Symbol,coupling::Real;massmod=0.1)
    isfinite(coupling) && isfinite(massmod) || throw(ArgumentError("nonfinite model parameter"))
    terms = name == :phi4 ? phi4_terms(coupling) :
            name == :shg ? sgordon_terms(coupling) :
            name == :sg ? sgordon_terms(im*coupling) :
            name == :phi4_m ? phi4_massmod_terms(coupling,massmod) :
            throw(ArgumentError("unknown excitation model: $name"))
    return ExcitationModel(name,filter(t->!iszero(t[1]),terms))
end
function ExcitationModel(s::State;massmod=nothing)
    isreal(s.model_param.coupling) || throw(ArgumentError("model coupling must be real"))
    s.model_param.name == "phi4_m" && isnothing(massmod) &&
        throw(ArgumentError("State metadata does not store massmod; supply ExcitationModel(s; massmod=...) explicitly"))
    ExcitationModel(Symbol(s.model_param.name),real(s.model_param.coupling);
                    massmod=isnothing(massmod) ? 0.1 : massmod)
end

struct ExcitationODEWorkspace
    pair::ExcitationPair
    spec::Observable
    operators::NTuple{3,Matrix{ComplexF64}}
    tmp::Matrix{ComplexF64}
end
function excitation_rhs!(du,u,ws,t)
    jac,j = 2abs(t),transformed_source(t)
    @views for k in 1:ws.spec.channels
        excitation_transfer!(du[:,:,:,k],u[:,:,:,k],ws.pair,ws.tmp)
        du[:,:,:,k] .*= jac
    end
    @views for edge in ws.spec.edges
        excitation_source_add!(du[:,:,:,edge.dst],u[:,:,:,edge.src],ws.operators,
                               edge.left,edge.coefficient*j)
    end
    return nothing
end
function excitation_observable(pair,spec;method=Vern7())
    s = pair.context.state
    p = s.integrator_param
    quadrature_check(p)
    a = sqrt(p.max_point)
    u0 = zeros(ComplexF64,s.D,s.D,4,spec.channels)
    u0[:,:,:,1] = pair.right
    ws = ExcitationODEWorkspace(pair,spec,excitation_operators(pair,spec.commutator),similar(s.R))
    prob = ODEProblem(excitation_rhs!,u0,(-a,a),ws)
    sol = solve(prob,method;save_start=false,save_end=true,save_everystep=false,dense=false,
                tstops=[0.0],abstol=p.atol,reltol=p.rtol,maxiters=p.maxiters)
    successful_retcode(sol) || throw(SolverFailure("excitation ODE failed: $(sol.retcode)"))
    all(isfinite,sol.u[end]) || throw(SolverFailure("nonfinite excitation trajectory"))
    return excitation_contract(pair,@view sol.u[end][:,:,:,end])
end

function excitation_kinetic_sum(pair,terms,n)
    result = 0.0im
    for (weight,spec) in terms
        ops = excitation_operators(pair,spec.commutator)
        for left in (false,true)
            b = zero(pair.right)
            excitation_source_add!(b,pair.right,ops,left,1)
            for (theta,w) in zip(kinetic_grid(n)...)
                x = excitation_resolvent(pair,b,cos(theta))
                y = zero(x)
                excitation_source_add!(y,x,ops,!left,1)
                result += (weight*w/2)*excitation_contract(pair,y)
            end
        end
    end
    return result
end
function excitation_kinetic(pair,terms,options)
    p = options
    p.minnodes>=2 && p.maxnodes>=2p.minnodes && isfinite(p.atol) && isfinite(p.rtol) &&
        p.atol>0 && p.rtol>=0 || throw(ArgumentError("invalid kinetic quadrature controls"))
    n = p.minnodes
    previous = excitation_kinetic_sum(pair,terms,n)
    while 2n<=p.maxnodes
        n *= 2
        current = excitation_kinetic_sum(pair,terms,n)
        isfinite(current) || throw(SolverFailure("nonfinite excitation kinetic quadrature"))
        abs(current-previous)<=p.atol+p.rtol*abs(current) && return current
        previous = current
    end
    throw(SolverFailure("excitation kinetic quadrature did not converge by $n nodes"))
end

function excitation_element(pair,model;kinetic_backend=:resolvent,
                            kinetic_options=KineticOptions(),method=Vern7())
    kinetic_backend in (:resolvent,:ode) || throw(ArgumentError("excitation kinetic backend must be :resolvent or :ode"))
    result = 0.0im
    kinetics = filter(t->t[2].kinetic,model.terms)
    if kinetic_backend == :resolvent && !isempty(kinetics)
        result += excitation_kinetic(pair,kinetics,kinetic_options)
    end
    for (weight,spec) in model.terms
        kinetic_backend == :resolvent && spec.kinetic && continue
        result += weight*excitation_observable(pair,spec;method)
    end
    isfinite(result) || throw(SolverFailure("nonfinite excitation matrix element"))
    return result
end

"""Connected Hamiltonian matrix element <Phi_p(u)|(H-E0)|Phi_p(w)> per
2πδ(0). u and w are horizontal W matrices (V=-R†W), not whitened coordinates.
This is a complex sesquilinear form, not a real energy Hessian.
"""
function excitation_matrix_element(s::State,p::Real,u::AbstractMatrix,w::AbstractMatrix;
                                   model=ExcitationModel(s),environment_tol=1e-10,max_bond=16,kw...)
    ctx = ExcitationContext(s,p;tol=environment_tol,max_bond)
    return excitation_element(ExcitationPair(ctx,u,w),model;kw...)
end

# W = X diag(rho_eigenvalues)^(-1/2) U† gives tr(W rho W†)=||X||².
# Resolve the metric by its support; never add the optimizer's metric regularizer.
function excitation_basis(s,metric_tol)
    isfinite(metric_tol) && 0<=metric_tol<1 || throw(ArgumentError("metric_tol must lie in [0,1)"))
    f = eigen(Hermitian(s.ρ0))
    keep = findall(>(metric_tol*maximum(f.values)),f.values)
    isempty(keep) && throw(SolverFailure("empty tangent metric support"))
    basis = Matrix{ComplexF64}[]
    for j in keep, i in 1:s.D
        w = zeros(ComplexF64,s.D,s.D)
        w[i,:] = conj.(f.vectors[:,j])/sqrt(f.values[j])
        push!(basis,w)
    end
    return basis,length(keep),sum(abs,f.values[setdiff(eachindex(f.values),keep)])
end

"""Build the dense Hamiltonian in an orthonormal complex tangent basis.
Use assembly=:action for O(D²) reverse-contracted actions, with adaptive ODEs and
quadrature as in the ground-state gradient. The default assembly=:bilinear checks
every entry independently, requiring O(D⁴) observable evaluations. Both retain
dense O(D⁶) transfer factorization and check Hermiticity before symmetrization.
Increase max_point and n_quad (action only), and tighten ODE tolerances to check
integration errors.
"""
function excitation_hamiltonian(s::State,p::Real;model=ExcitationModel(s),metric_tol=1e-12,
                                environment_tol=1e-10,hermitian_tol=1e-7,max_bond=16,
                                assembly=:bilinear,kw...)
    isfinite(hermitian_tol) && hermitian_tol>0 || throw(ArgumentError("invalid Hermiticity tolerance"))
    assembly in (:bilinear,:action) || throw(ArgumentError("assembly must be :bilinear or :action"))
    ctx = ExcitationContext(s,p;tol=environment_tol,max_bond)
    basis,rank,discarded = excitation_basis(ctx.state,metric_tol)
    n = length(basis)
    h = Matrix{ComplexF64}(undef,n,n)
    for j in 1:n
        if assembly == :action
            action = excitation_hamiltonian_action(ctx,basis[j];model,kw...)
            for i in 1:n
                h[i,j] = dot(basis[i],action)
            end
        else
            for i in 1:n
                h[i,j] = excitation_element(ExcitationPair(ctx,basis[i],basis[j]),model;kw...)
            end
        end
    end
    defect = norm(h-h')/max(1,norm(h))
    defect <= hermitian_tol || throw(SolverFailure("excitation Hamiltonian Hermiticity defect $defect exceeds $hermitian_tol"))
    return (hamiltonian=Hermitian((h+h')/2),basis=basis,momentum=Float64(p),
            hermiticity_error=defect,metric_rank=rank,discarded_metric_weight=discarded,
            model=model.name)
end

"""Lowest tangent-space Ritz energies at a fixed momentum, with normalized W
vectors and algebraic residuals. Negative energies are returned unchanged: they
can signal an inadequate background. Optimize the ground state first and check
bond dimension and numerical convergence before interpreting levels as particles.
Only the sector with the same vacuum at both spatial ends is included.
"""
function excitation_spectrum(s::State,p::Real;nev::Integer=1,kw...)
    nev>0 || throw(ArgumentError("nev must be positive"))
    result = excitation_hamiltonian(s,p;kw...)
    n = size(result.hamiltonian,1)
    nev<=n || throw(ArgumentError("nev=$nev exceeds the tangent dimension $n"))
    f = eigen(result.hamiltonian)
    energies = f.values[1:nev]
    coefficients = f.vectors[:,1:nev]
    vectors = [sum(coefficients[i,j]*result.basis[i] for i in 1:n) for j in 1:nev]
    residuals = [norm(result.hamiltonian*coefficients[:,j]-energies[j]*coefficients[:,j]) for j in 1:nev]
    return merge(result,(energies=energies,vectors=vectors,coefficients=coefficients,residuals=residuals))
end

"""Lowest p=0 tangent-space energy (the rest gap in the represented sector).
Unlike the legacy mass_gap(s), this uses the Hamiltonian. This does not search for
nonzero-momentum minima or topological excitations between distinct vacua.
"""
excitation_gap(s::State;kw...) = first(excitation_spectrum(s,0.;nev=1,kw...).energies)
mass_gap(s::State,model::ExcitationModel;kw...) = excitation_gap(s;model,kw...)

"""Compute independently energy-sorted Ritz levels at each requested momentum.
Columns need not follow a particle through a crossing; inspect vectors/symmetries.
"""
function dispersion_relation(s::State,momenta;nev::Integer=1,kw...)
    ps = Float64.(collect(momenta))
    !isempty(ps) && all(isfinite,ps) || throw(ArgumentError("momenta must be nonempty and finite"))
    results = [excitation_spectrum(s,p;nev,kw...) for p in ps]
    energies = permutedims(hcat((r.energies for r in results)...))
    return (momenta=ps,energies=energies,spectra=results)
end

"""For one energy-sorted branch, report E(p)²-target_speed²*p² and the least-squares fit
E² = intercept + c_squared*p². Requires at least two distinct p². Fit diagnostics
are descriptive; they do not establish Lorentz invariance or identify branches.
target_speed defaults to 1 for the mass-one models; adjust for rescaled units.
"""
function relativistic_diagnostics(dispersion;branch::Integer=1,target_speed::Real=1)
    isfinite(target_speed) && target_speed>0 || throw(ArgumentError("target_speed must be finite and positive"))
    e = dispersion.energies
    1<=branch<=size(e,2) || throw(ArgumentError("invalid branch"))
    ps = dispersion.momenta
    length(unique(abs2.(ps)))>=2 || throw(ArgumentError("at least two distinct squared momenta are required"))
    energies = e[:,branch]
    all(>=(0),energies) || throw(ArgumentError("negative excitation energies cannot be interpreted as a relativistic branch"))
    p2,y = abs2.(ps),abs2.(energies)
    a = hcat(ones(length(ps)),p2)
    intercept,c2 = a\y
    invariant = y-target_speed^2*p2
    return (invariant_mass_squared=invariant,intercept=intercept,c_squared=c2,
            fit_residual=norm(y-a*[intercept,c2])/max(norm(y),eps()),
            invariant_spread=maximum(invariant)-minimum(invariant))
end
