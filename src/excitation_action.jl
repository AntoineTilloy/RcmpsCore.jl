# Reverse contraction of the bra insertion in excitation_environments.jl.
# Returns F(W) with H(U,W)=tr(U† F(W)); it is NOT an energy Hessian.
struct ExcitationActionWorkspace{Adjoint}
    context::ExcitationContext
    w::Matrix{ComplexF64}
    v::Matrix{ComplexF64}
    spec::Observable
    c::Matrix{ComplexF64}
    cw::Matrix{ComplexF64}
    tmp::Matrix{ComplexF64}
end

function action_operators(ctx,w,v,commutator)
    s=ctx.state
    commutator || return s.R,w
    return s.Q*s.R-s.R*s.Q,s.Q*w-w*s.Q+v*s.R-s.R*v-im*ctx.momentum*w
end

function action_rhs!(du,u,ws::ExcitationActionWorkspace{adj},t) where adj
    s,p=ws.context.state,ws.context.momentum
    jac,j=2abs(t),transformed_source(t)
    @views for k in 1:ws.spec.channels
        x,y=u[:,:,1,k],u[:,:,2,k]
        dx,dy=du[:,:,1,k],du[:,:,2,k]
        if adj
            lindblad_adjoint!(dx,s,x,ws.tmp);lindblad_adjoint!(dy,s,y,ws.tmp)
            dx .+= im*p .* x
            mul!(dx,y,ws.v,1,1)
            addtriple!(dx,s.R',y,ws.w,1,ws.tmp)
        else
            lindblad!(dx,s,x,ws.tmp);lindblad!(dy,s,y,ws.tmp)
            dy .-= im*p .* y
            mul!(dy,ws.v,x,1,1)
            addtriple!(dy,ws.w,x,s.R',1,ws.tmp)
        end
        dx .*= jac;dy .*= jac
    end
    @views for edge in ws.spec.edges
        a=edge.coefficient*j
        src,dst=adj ? (edge.dst,edge.src) : (edge.src,edge.dst)
        x,y=u[:,:,1,src],u[:,:,2,src]
        dx,dy=du[:,:,1,dst],du[:,:,2,dst]
        if adj && edge.left
            mul!(dx,x,ws.c,a,1);mul!(dx,y,ws.cw,a,1);mul!(dy,y,ws.c,a,1)
        elseif adj
            mul!(dx,ws.c',x,a,1);mul!(dy,ws.c',y,a,1)
        elseif edge.left
            mul!(dx,ws.c,x,a,1);mul!(dy,ws.c,y,a,1);mul!(dy,ws.cw,x,a,1)
        else
            mul!(dx,x,ws.c',a,1);mul!(dy,y,ws.c',a,1)
        end
    end
    return nothing
end

function action_trajectory(ctx,w,v,rw,spec,points,adj,method)
    s=ctx.state;p=s.integrator_param;a=sqrt(p.max_point)
    u0=zeros(ComplexF64,s.D,s.D,2,spec.channels)
    if adj
        u0[:,:,2,end]=Matrix{ComplexF64}(I,s.D,s.D)
    else
        u0[:,:,1,1]=s.ρ0;u0[:,:,2,1]=rw
    end
    c,cw=action_operators(ctx,w,v,spec.commutator)
    ws=ExcitationActionWorkspace{adj}(ctx,w,v,spec,c,cw,similar(w))
    sol=solve(ODEProblem(action_rhs!,u0,(-a,a),ws),method;
        saveat=vcat(points,a),save_start=false,save_end=true,save_everystep=false,
        dense=false,tstops=[0.],abstol=p.atol,reltol=p.rtol,maxiters=p.maxiters)
    successful_retcode(sol) || throw(SolverFailure("excitation action ODE failed: $(sol.retcode)"))
    all(isfinite,sol.u[end]) || throw(SolverFailure("nonfinite excitation action trajectory"))
    return sol
end

bra_transfer_pull(s,a,x)=(a*s.R-s.R*a)*x
ket_trace_dual(s,v,w,a)=a*v+s.R'*a*w
function bra_source_pull(ctx,x,commutator)
    s=ctx.state
    commutator || return x
    return s.Q'*x-x*s.Q'-s.R*x*s.R'+s.R*s.R'*x+im*ctx.momentum*x
end
function action_dual_poisson(ctx,b;shifted=false)
    s=ctx.state
    rhs=b-tr(s.ρ0*b)*I
    x=reshape((shifted ? ctx.ketfactor' : ctx.poisson')\vec(rhs),s.D,s.D)
    x-=tr(s.ρ0*x)*I
    residual=-lindblad_adjoint(s,x)-(shifted ? im*ctx.momentum*x : zero(x))-rhs
    norm(residual)<=20ctx.tol*max(1,norm(rhs)) || throw(SolverFailure("excitation action boundary residual too large"))
    return x
end
function action_boundaries(ctx,w,v,rw,a0,a1,xout)
    s=ctx.state
    d=action_dual_poisson(ctx,a1)
    e=action_dual_poisson(ctx,a0+ket_trace_dual(s,v,w,d);shifted=true)
    return bra_transfer_pull(s,e,s.ρ0)+bra_transfer_pull(s,d,rw)+d*w*s.ρ0+
           w*excitation_poisson(ctx,xout)
end

function excitation_observable_action(ctx,w,v,rw,spec;method=Vern7(),parallel=false)
    s=ctx.state
    points,weights=quadrature(s.integrator_param)
    forward,backward=evaluation_map((false,true),parallel) do adj
        action_trajectory(ctx,w,v,rw,spec,points,adj,method)
    end
    out=zero(w);wrho=w*s.ρ0
    @views for i in eachindex(points)
        x,a=forward.u[i],backward.u[length(points)-i+1]
        dx,j=weights[i]*2abs(points[i]),weights[i]*transformed_source(points[i])
        for k in 1:spec.channels
            x0,x1=x[:,:,1,k],x[:,:,2,k]
            a0,a1=a[:,:,1,k],a[:,:,2,k]
            out .+= dx.*(bra_transfer_pull(s,a0,x0)+bra_transfer_pull(s,a1,x1)+
                          a1*w*x0-tr(a1*x0)*wrho)
        end
        for edge in spec.edges
            edge.left && continue
            b=a[:,:,1,edge.dst]*x[:,:,1,edge.src]+a[:,:,2,edge.dst]*x[:,:,2,edge.src]
            out .+= (j*edge.coefficient).*bra_source_pull(ctx,b,spec.commutator)
        end
    end
    @views out .+= action_boundaries(ctx,w,v,rw,backward.u[end][:,:,1,1],
                    backward.u[end][:,:,2,1],forward.u[end][:,:,1,end])
    return out
end

action_factor(ctx,z;shifted=false)=resolvent_factor(ctx,z,shifted ? 1 : 0)
function action_resolve(ctx,b,z;shifted=false,adj=false,factor=nothing)
    f=ctx.schur;s=ctx.state
    isnothing(factor) && (factor=action_factor(ctx,z;shifted))
    return reshape(f.Z*((adj ? factor' : factor)\(f.Z'*vec(b))),s.D,s.D)
end

function excitation_kinetic_action_sum(ctx,w,v,rw,terms,n)
    s=ctx.state;out=zero(w);wrho=w*s.ρ0
    grid=kinetic_grid(n)
    # The boundary pullback is linear: accumulate every term and order, then solve once.
    total_a0,total_a1,total_x0=zero(w),zero(w),zero(w)
    prepared=[(coefficient,spec,action_operators(ctx,w,v,spec.commutator)...) for (coefficient,spec) in terms]
    for (theta,nodeweight) in zip(grid...)
        z=cos(theta)
        # One shifted matrix per node serves every term, order, forward and reverse solve.
        # At p=0 all four solves share it. Do not retain one per grid point.
        factor=action_factor(ctx,z)
        shiftedfactor=iszero(ctx.momentum) ? factor : action_factor(ctx,z;shifted=true)
        for (coefficient,spec,c,cw) in prepared
            orders=spec.kinetic ? ((false,true),(true,false)) :
                                 ((false,false),(false,true),(true,false),(true,true))
            weight=nodeweight*coefficient/2
            for (left,secondleft) in orders
                # First source and forward resolvent, retaining the ground and ket blocks.
                b0=left ? c*s.ρ0 : s.ρ0*c'
                b1=left ? c*rw+cw*s.ρ0 : rw*c'
                x0=action_resolve(ctx,b0,z;factor)
                x1=action_resolve(ctx,b1+z*ket_insertion(s,v,w,x0),z;shifted=true,factor=shiftedfactor)
                xout=secondleft ? c*x0 : x0*c'
                # Reverse second source; the terminal mixed-block covector is I.
                if !secondleft # second source is a bra source
                    a0,a1=zero(w),Matrix(c')
                    out .+= weight.*bra_source_pull(ctx,x1,spec.commutator)
                else
                    a0,a1=copy(cw),copy(c)
                end
                a1=action_resolve(ctx,a1,z;adj=true,factor)
                a0=action_resolve(ctx,a0+z*ket_trace_dual(s,v,w,a1),z;shifted=true,adj=true,factor=shiftedfactor)
                out .+= (weight*z).*(bra_transfer_pull(s,a0,x0)+bra_transfer_pull(s,a1,x1)+
                                      a1*w*x0-tr(a1*x0)*wrho)
                # Reverse first source and accumulate the two boundary covectors.
                if left
                    a0,a1=a0*c+a1*cw,a1*c
                else
                    out .+= weight.*bra_source_pull(ctx,a0*s.ρ0+a1*rw,spec.commutator)
                    a0,a1=c'*a0,c'*a1
                end
                total_a0 .+= weight.*a0;total_a1 .+= weight.*a1;total_x0 .+= weight.*xout
            end
        end
    end
    out .+= action_boundaries(ctx,w,v,rw,total_a0,total_a1,total_x0)
    return out
end
function excitation_kinetic_action(ctx,w,v,rw,terms,options)
    p=options
    p.minnodes>=2 && p.maxnodes>=2p.minnodes && isfinite(p.atol) && isfinite(p.rtol) &&
        p.atol>0 && p.rtol>=0 || throw(ArgumentError("invalid kinetic quadrature controls"))
    n=p.minnodes;previous=excitation_kinetic_action_sum(ctx,w,v,rw,terms,n)
    while 2n<=p.maxnodes
        n*=2;current=excitation_kinetic_action_sum(ctx,w,v,rw,terms,n)
        all(isfinite,current) || throw(SolverFailure("nonfinite kinetic action"))
        norm(current-previous)<=p.atol+p.rtol*norm(current) && return current
        previous=current
    end
    throw(SolverFailure("excitation kinetic action quadrature did not converge by $n nodes"))
end

"""Apply the Hamiltonian by reverse contraction of the bra tangent, returning F
such that excitation_matrix_element(s,p,U,W) == dot(U,F). Coordinates here are
unwhitened W matrices. Dense D² transfer factors are retained, but no D⁴ set of
bilinear evaluations is needed. Potential contractions use n_quad quadrature.
With `kinetic_nodes=nothing` the kinetic θ grid is refined adaptively for this W;
an integer fixes it, which makes the kinetic part exactly linear in W (adaptive ODE
potential terms are linear to their tolerance), as an iterative eigensolver requires.
Check a fixed grid by refinement.
"""
function excitation_hamiltonian_action(ctx::ExcitationContext,w::AbstractMatrix;
        model=ExcitationModel(ctx.state),kinetic_backend=:resolvent,
        kinetic_options=KineticOptions(),kinetic_nodes=nothing,method=Vern7(),parallel=:serial)
    s=ctx.state
    size(w)==size(s.R) || throw(DimensionMismatch("tangent dimensions do not match the state"))
    all(isfinite,w) || throw(ArgumentError("nonfinite tangent"))
    kinetic_backend in (:resolvent,:ode) || throw(ArgumentError("unknown excitation kinetic backend"))
    check_parallel(parallel)
    isnothing(kinetic_nodes) || check_kinetic_nodes(kinetic_nodes)
    w=Matrix{ComplexF64}(w);v=-s.R'*w
    rw=excitation_poisson(ctx,ket_insertion(s,v,w,s.ρ0);shift=1)
    isquadratic(spec)=spec.channels==3 && !spec.kinetic &&
        spec.edges==monomial_spec(2).edges
    resolved(spec)=spec.kinetic || isquadratic(spec)
    kinetics=filter(t->resolved(t[2]),model.terms)
    potentials=filter(t->kinetic_backend==:ode || !resolved(t[2]),model.terms)
    jobs=collect(1:length(potentials)+(kinetic_backend==:resolvent && !isempty(kinetics)))
    results=evaluation_map(jobs,parallel in (:terms,:both)) do i
        if i>length(potentials)
            isnothing(kinetic_nodes) ? excitation_kinetic_action(ctx,w,v,rw,kinetics,kinetic_options) :
                excitation_kinetic_action_sum(ctx,w,v,rw,kinetics,kinetic_nodes)
        else
            c,spec=potentials[i]
            c.*excitation_observable_action(ctx,w,v,rw,spec;method,
                                           parallel=parallel in (:trajectories,:both))
        end
    end
    out=isempty(results) ? zero(w) : reduce(+,results)
    all(isfinite,out) || throw(SolverFailure("nonfinite Hamiltonian action"))
    return out
end
function excitation_hamiltonian_action(s::State,p::Real,w::AbstractMatrix;
        environment_tol=1e-10,max_bond=16,kw...)
    excitation_hamiltonian_action(ExcitationContext(s,p;tol=environment_tol,max_bond),w;kw...)
end
