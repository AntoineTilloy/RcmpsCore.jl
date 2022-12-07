#We need to use a slightly modified version of the OptimKit.jl module from Ghent. 
#The only modification is to allow to input step sizes for GradientDescent
#in the LBFGS algorithm. Without it, the algorithm tries steps that are too large 
#in the first few iterations
include("OptimKit_modified/OptimKit.jl")



function metric(state::State,W1,W2)
    ρ0_reg=state.ρ0 + state.manifold_param.metric_reg*I
    g=2*real(tr(W1*ρ0_reg*W2'))
    return g
end

function finalize!(x, f, g, numiter)
    innergg = metric(x, g, g)
    x.manifold_param.precond_reg = max(10*eps(eltype(real(x.R))),
        x.manifold_param.precond_reg_prefactor * min(innergg, 1.0))

    return x, f, g
end

function precondition(state::State, W)
    ρ0_reg1 = state.ρ0 + state.manifold_param.precond_reg * I
    ρ0_reg2 = state.ρ0 + state.manifold_param.metric_reg * I
    return W * ρ0_reg2 * inv(ρ0_reg1)
end

function retraction(state::State,W::AbstractMatrix,α::Number)
    Kt=state.K + 0.5*1im*((W')*(α*state.R+α^2/2*W) -  (α*state.Rd+α^2/2*W')*W)
    Rt=state.R + α*W
    newstate=State(Rt,Kt)
    copy_param!(newstate,state)
    return newstate,W
end

function retraction_regulated(state::State,W::AbstractMatrix,alpha::Number;regulator=1e0)
    normgrad=sqrt(metric(state,W,W))
    if alpha*normgrad<regulator
        α=alpha
    else
        α=(regulator/normgrad)*(alpha*normgrad/regulator)^0.2
    end
    newstate,W=retraction(state,W,α)
    return newstate,W
end

function naive_transport!(ξ_in,state_in::State,W_in,alpha,state_out)
    ξ_in[:,:]=ξ_in
end

function optimize_state(state::State;
    method="GD",
    n_iter=1000,
    gradtol=1e-7,
    retract=retraction_regulated,
    transport! = naive_transport!,
    precond = precondition,
    inner = metric,
    mlbfgs=100,
    verbosity=2,
    ls = ls_lenient)

    ls_current = ls
    model = state.model_param.name
    coupling = state.model_param.coupling

    if model=="sg"
            fg=state->fg_sgordon(state,1im*coupling)
    elseif model=="shg" 
            fg=state->fg_sgordon(state,coupling)
    elseif model=="phi4"
            fg=state->fg_phi4(state,real(coupling))
    end

    if method=="GD"
        optimizer=GradientDescent(verbosity=verbosity,maxiter=n_iter,gradtol=gradtol,linesearch=ls_current)
    elseif method=="LBFGS"
        optimizer=LBFGS(mlbfgs,verbosity=verbosity,maxiter=n_iter,gradtol=gradtol,linesearch = ls_current)
    else 
        optimizer=ConjugateGradient(verbosity=verbosity,maxiter=n_iter,gradtol=gradtol,linesearch=ls_current)
    end
    
    state, e0, grad, numfg, normgradhistory = optimize(
            fg,
            state,
            optimizer;
            inner = inner,
            transport! = transport!,
            finalize! = finalize!,
            retract = retract,
            precondition = precond)
    state.e0=e0
    state.grad=grad
    return state
end

#Two typical parameters for the line search: the standard one and a more lenient one (that accepts steps more easily)

ls_default=HagerZhangLineSearch(c₁= .1, # parameter for first Wolfe condition
                        c₂ = .9, # paremeter for second Wolfe condition
                        ϵ = 1e-6, # parameter for approximate Wolfe condition, accept fluctation of ϵ on the function value
                        θ = 1/2, # used in bisection
                        γ = 2/3, # determines when a bisection step is performed
                        ρ = 5. # expansion parameter for initial bracket interval
                        )

ls_lenient=HagerZhangLineSearch(c₁= 1e-2, # parameter for first Wolfe condition
                        c₂ =0.99, # paremeter for second Wolfe condition
                        ϵ = 1e-5, # parameter for approximate Wolfe condition, accept fluctation of ϵ on the function value
                        θ = 1/2, # used in bisection
                        γ = 2/3, # determines when a bisection step is performed
                        ρ = 5.0 # expansion parameter for initial bracket interval
                        )





