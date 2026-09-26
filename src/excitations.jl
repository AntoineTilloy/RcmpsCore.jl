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
    sources = map(terms) do (weight,spec)
        ops = excitation_operators(pair,spec.commutator)
        bs = map((false,true)) do left
            b = zero(pair.right)
            excitation_source_add!(b,pair.right,ops,left,1)
        end
        (weight,ops,bs)
    end
    # The shifted triangular factors depend only on the node: form them once for all terms.
    for (theta,w) in zip(kinetic_grid(n)...)
        z = cos(theta)
        factors = resolvent_factors(pair.context,z)
        for (weight,ops,bs) in sources, (left,b) in zip((false,true),bs)
            x = excitation_resolvent(pair,b,z;factors)
            y = zero(x)
            excitation_source_add!(y,x,ops,!left,1)
            result += (weight*w/2)*excitation_contract(pair,y)
        end
    end
    return result
end
function check_kinetic_nodes(n)
    n isa Integer && n >= 2 || throw(ArgumentError("kinetic_nodes must be an integer of at least 2"))
    return Int(n)
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

# kinetic_nodes=nothing refines the kinetic θ grid adaptively; an integer fixes it.
function excitation_element(pair,model;kinetic_backend=:resolvent,
                            kinetic_options=KineticOptions(),kinetic_nodes=nothing,method=Vern7())
    kinetic_backend in (:resolvent,:ode) || throw(ArgumentError("excitation kinetic backend must be :resolvent or :ode"))
    result = 0.0im
    kinetics = filter(t->t[2].kinetic,model.terms)
    if kinetic_backend == :resolvent && !isempty(kinetics)
        result += isnothing(kinetic_nodes) ? excitation_kinetic(pair,kinetics,kinetic_options) :
            excitation_kinetic_sum(pair,kinetics,check_kinetic_nodes(kinetic_nodes))
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

# Orthonormal basis W_k with tr(W_k rho W_l†)=δ_kl (see TangentCoordinates: Cholesky
# whitening when the metric is fully resolved, eigen-whitening on its support otherwise).
# Resolve the metric by its support; never add the optimizer's metric regularizer.
function excitation_basis(s,metric_tol)
    c = TangentCoordinates(s;sector=:all,parity=nothing,gauge=:original,metric_tol)
    return tangent_basis(c),c.metric_rank,c.discarded_metric_weight
end

"""Build the dense Hamiltonian in an orthonormal complex tangent basis.
Use assembly=:action for O(D²) reverse-contracted actions, with adaptive ODEs and
quadrature as in the ground-state gradient. The default assembly=:bilinear checks
every entry independently, requiring O(D⁴) observable evaluations. Both retain
dense O(D⁶) transfer factorization and check Hermiticity before symmetrization.
Increase max_point and n_quad (action only), and tighten ODE tolerances to check
integration errors.

`sector=:odd` or `:even` restricts to one physical parity sector of a Z₂-symmetric
state (see `TangentCoordinates`); the default `:all` uses the full tangent space.
`gauge=:density` evaluates in the density eigenbasis. The basis is Cholesky-whitened
when the metric is fully resolved (eigen-whitened on its support otherwise), so the
matrix and coefficients are basis-dependent while the spectrum is not. Returned basis
matrices are always in the gauge of `s`. For large D use `excitation_eigsolve` instead.
"""
function excitation_hamiltonian(s::State,p::Real;model=ExcitationModel(s),metric_tol=1e-12,
                                environment_tol=1e-10,hermitian_tol=1e-7,max_bond=16,
                                assembly=:bilinear,sector=:all,
                                parity=(sector == :all ? nothing : :auto),parity_tol=1e-12,
                                gauge=:original,kw...)
    isfinite(hermitian_tol) && hermitian_tol>0 || throw(ArgumentError("invalid Hermiticity tolerance"))
    assembly in (:bilinear,:action) || throw(ArgumentError("assembly must be :bilinear or :action"))
    isfinite(p) || throw(ArgumentError("invalid momentum or environment tolerance"))
    s.D <= max_bond || throw(ArgumentError("dense excitation reference is limited to D≤$max_bond; increase max_bond explicitly to accept its cost"))
    coordinates = TangentCoordinates(s;sector,parity,parity_tol,gauge,metric_tol)
    ctx = ExcitationContext(coordinates.state,p;tol=environment_tol,max_bond)
    basis = tangent_basis(coordinates)
    rank,discarded = coordinates.metric_rank,coordinates.discarded_metric_weight
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
    return (hamiltonian=Hermitian((h+h')/2),basis=[original_tangent(coordinates,w) for w in basis],
            momentum=Float64(p),hermiticity_error=defect,metric_rank=rank,
            discarded_metric_weight=discarded,model=model.name,sector=coordinates.sector)
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

function check_solver(solver)
    solver in (:dense,:iterative) || throw(ArgumentError("solver must be :dense or :iterative"))
    return solver
end

"""Lowest p=0 tangent-space energy (the rest gap in the represented sector).
Unlike the legacy mass_gap(s), this uses the Hamiltonian. This does not search for
nonzero-momentum minima or topological excitations between distinct vacua.

`solver=:dense` (default) diagonalizes the full dense matrix (small-D reference, full
tangent space, adaptive kinetic quadrature, standard integration settings).
`solver=:iterative` calls `excitation_eigsolve`: the default sector of a Z₂-symmetric
state is the odd one, the kinetic grid is fixed, and the validated (refined bilinear)
energy is returned. A `SolverFailure` is thrown unless that level is `accepted`.
See `dispersion_relation` for the full list of differing defaults.
"""
function excitation_gap(s::State;solver=:dense,kw...)
    check_solver(solver) == :dense && return first(excitation_spectrum(s,0.;nev=1,kw...).energies)
    return first(accepted_energies(excitation_eigsolve(s,0.;nev=1,kw...)))
end
mass_gap(s::State,model::ExcitationModel;kw...) = excitation_gap(s;model,kw...)

"""Compute independently energy-sorted Ritz levels at each requested momentum.
Columns need not follow a particle through a crossing; inspect vectors/symmetries.
With `solver=:iterative`, the momentum-independent factors are computed once, each
momentum starts from the previous eigenvectors, and one parity sector is used
(odd by default for Z₂-symmetric states); energies are then the validated ones. By
default (`strict=true`) a `SolverFailure` is thrown at the first momentum whose levels
are not `accepted`; `strict=false` computes every momentum and returns an `accepted`
vector instead. The two solvers also differ in their defaults: dense uses the full
tangent space, the original gauge, `metric_tol=1e-12`, adaptive kinetic quadrature and
`max_bond=16`; iterative uses one sector, the density gauge, `metric_tol=0` above a
10D·eps floor, 256 fixed kinetic nodes and `max_bond=64`.
"""
function dispersion_relation(s::State,momenta;nev::Integer=1,solver=:dense,strict=true,kw...)
    ps = Float64.(collect(momenta))
    !isempty(ps) && all(isfinite,ps) || throw(ArgumentError("momenta must be nonempty and finite"))
    if check_solver(solver) == :dense
        results = [excitation_spectrum(s,p;nev,kw...) for p in ps]
        return (momenta=ps,energies=permutedims(hcat((r.energies for r in results)...)),spectra=results)
    end
    results = iterative_dispersion(s,ps;nev,strict,kw...)
    levels = [strict ? accepted_energies(r) : reported_energies(r) for r in results]
    all(l->length(l) == nev,levels) ||
        throw(SolverFailure("some momenta returned fewer than nev=$nev levels; see spectra"))
    return (momenta=ps,energies=permutedims(hcat(levels...)),spectra=results,
            accepted=[r.accepted for r in results])
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
