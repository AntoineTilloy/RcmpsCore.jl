using OrdinaryDiffEq
using Distributed
import Optim

#Compute expectation values and gradient of expectation values of typical
#local operators (field monomials, vertex operators, kinetic terms)

#Gradients are computed via backpropagation using a tanhsinh quadrature
#to save (a lot) of memory

dt = 1e-7
maxiter = 10^7
dtmin = 1e-15

# function generate_tanhsinh_weights(param::IntegratorParameters)
#     n=param.n_tanhsinh
#     overshoot = param.overshoot
#     tbound = param.t_bound

#     h=2.0^(-n)
#     N=floor(Int,1/h*overshoot)

#     u=LinRange(-N,N,2*N+1)
#     x=tanh.(0.5*π*sinh.(h*u))
#     weight=0.5*h*π*cosh.(h*u)./(cosh.(0.5*π*sinh.(h*u)).^2)*tbound/2
#     #Negative side of the interval
#     tneg=(x .- 1)*tbound*0.5
#     tpos=(x .+ 1)*tbound*0.5

#     points=vcat(tneg,tpos)
#     weights=vcat(weight,weight)

#     return points,weights
# end




function vertex(state::State,β::Number;method=Vern7())
    abstol=state.integrator_param.atol
    reltol=state.integrator_param.rtol
    points,weights=symmetric_exp_sinh_quadrature(state.integrator_param)
    tbound=points[end]
    tspan=(-tbound,tbound)
    
    #Solve the ODE for the quadrature points
    p=[state::State,β::Number]
    #Solve the forward problem
    prob_forward = ODEProblem(vertex_generator!,state.ρ0,tspan,p)
    ρ = solve(prob_forward,method,saveat=points,reltol=reltol,abstol=abstol,dt=dt,dtmin=dtmin,force_dtmin=true,maxiters=maxiter)


    #Solve the backward problem
    id=one(state.R)
    prob_backward = ODEProblem(vertex_generator_adjoint!,id,tspan,p)
    σ =  solve(prob_backward,method,saveat=points,reltol=reltol,abstol=abstol,dt=dt,dtmin=dtmin,force_dtmin=true,maxiters=maxiter)

    ρ=fetch(ρ) ; σ=fetch(σ)
    A=zero(state.R) ; B=zero(state.R) ; C=zero(state.R)

    for i in 1:size(points)[1]
        t=points[i]
        w=weights[i]*f_prime(t)
        A+=w*ρ[i]*state.Rd*σ[end-i+1]
        B+=w*β*jsource(f(t))*ρ[i]*σ[end-i+1]
        C+=-w*ρ[i]*σ[end-i+1]*state.Rd
    end

    M=A+B+C

    #grad=adjoint(robust_inv(state.ρ0,state.manifold_param.metric_reg)*M)
    grad=adjoint(robust_inv(state.ρ0,M,state.manifold_param.metric_reg))
    vert=tr(ρ(tbound))

    return vert,grad
end




function phi_monomial(state::State,n::Integer;method=Vern7())
    abstol=state.integrator_param.atol
    reltol=state.integrator_param.rtol
    points,weights=symmetric_exp_sinh_quadrature(state.integrator_param)
    tbound=points[end]
    tspan=(-tbound,tbound)

    p=[state,n]
    D=state.D
    
    #Solve the forward problem
    ρ0=zeros(eltype(state.R),n+1,D,D)
    ρ0[1,:,:]=state.ρ0
    prob_forward = ODEProblem(phi_generator!,ρ0,tspan,p)
    ρ = solve(prob_forward,method,saveat=points,reltol=reltol,abstol=abstol,dt=dt,dtmin=dtmin,force_dtmin=true,maxiters=maxiter)


    #Solve the backward problem
    σ0=zeros(eltype(state.R),n+1,D,D)
    id=one(state.R)
    σ0[n+1,:,:]=id
    prob_backward = ODEProblem(phi_generator_adjoint!,σ0,tspan,p)
    σ = solve(prob_backward,method,saveat=points,reltol=reltol,abstol=abstol,dt=dt,dtmin=dtmin,force_dtmin=true,maxiters=maxiter)

    ρ=fetch(ρ) ; σ=fetch(σ)

    M=zero(state.R)
    for i in 1:size(points)[1]
        t=points[i]
        rho=ρ[i]
        sigma=σ[end-i+1]
        w=weights[i]*f_prime(t)
        for k in 1:(n+1)
            M+=w*rho[k,:,:]*state.Rd*sigma[k,:,:]
            M+= -w*rho[k,:,:]*sigma[k,:,:]*state.Rd
        end
        for k in 2:(n+1)
            M+=w*jsource(f(t))*rho[k-1,:,:]*sigma[k,:,:]*(k-1)
        end
    end

    #grad=adjoint(robust_inv(state.ρ0,state.manifold_param.metric_reg)*M)
    grad=adjoint(robust_inv(state.ρ0,M,state.manifold_param.metric_reg))
    phi_n=real(tr(ρ(tbound)[n+1,:,:]))

    return phi_n,grad
end


function kinetic1(state::State;method=Vern7())
    abstol=state.integrator_param.atol
    reltol=state.integrator_param.rtol
    points,weights=symmetric_exp_sinh_quadrature(state.integrator_param)
    tbound=points[end]
    tspan=(-tbound,tbound)
    p=state
    D=state.D

    #Solve the forward problem
    ρ0=zeros(eltype(state.R),3,D,D)
    prob_forward = ODEProblem(kinetic1_generator!,ρ0,tspan,p)
    ρ = solve(prob_forward,method,saveat=points,reltol=reltol,abstol=abstol,dt=dt,dtmin=dtmin,force_dtmin=true,maxiters=maxiter)

    #Solve the backward problem
    σ0=zeros(eltype(state.R),3,D,D)
    prob_backward = ODEProblem(kinetic1_generator_adjoint!,σ0,tspan,p)
    σ = solve(prob_backward,method,saveat=points,reltol=reltol,abstol=abstol,dt=dt,dtmin=dtmin,force_dtmin=true,maxiters=maxiter)

    ρ=fetch(ρ) ; σ=fetch(σ)
    M=zero(state.R)
    for i in 1:size(points)[1]
        t=points[i]
        J=sqrt(2)*jsource(f(t))
        w=weights[i]*f_prime(t)
        rho=ρ[i]
        sigma=σ[end-i+1]

        #order 0
        M+=w*state.ρ0*state.Rd*sigma[1,:,:]
        M+= - w*state.ρ0*sigma[1,:,:]*state.Rd
        #order 1
        M+=w*rho[1,:,:]*state.Rd*sigma[2,:,:] +w*J*state.ρ0*sigma[2,:,:]
        M+=- w*rho[1,:,:]*sigma[2,:,:]*state.Rd
        #order 2
        M+=w*rho[2,:,:]*state.Rd*sigma[3,:,:]
        M+=- w*rho[2,:,:]*sigma[3,:,:]*state.Rd
        #order3
        M+=w*rho[3,:,:]*state.Rd + w*J*rho[2,:,:]
        M+= -w*rho[3,:,:]*state.Rd

    end

    #grad=adjoint(robust_inv(state.ρ0,state.manifold_param.metric_reg)*M)
    grad=adjoint(robust_inv(state.ρ0,M,state.manifold_param.metric_reg))
    kin1=real(tr(ρ(tbound)[3,:,:]))

    return kin1,grad
end


function kinetic2(state::State;method=Vern7())
    abstol=state.integrator_param.atol
    reltol=state.integrator_param.rtol
    points,weights=symmetric_exp_sinh_quadrature(state.integrator_param)
    tbound=points[end]
    tspan=(-tbound,tbound)
    p=state
    D=state.D

    #Solve the forward problem
    ρ0=zeros(eltype(state.R),3,D,D)
    prob_forward = ODEProblem(kinetic2_generator!,ρ0,tspan,p)
    ρ = solve(prob_forward,method,saveat=points,reltol=reltol,abstol=abstol,dt=dt,dtmin=dtmin,force_dtmin=true,maxiters=maxiter)

    #Solve the backward problem
    σ0=zeros(eltype(state.R),3,D,D)
    prob_backward = ODEProblem(kinetic2_generator_adjoint!,σ0,tspan,p)
    σ = solve(prob_backward,method,saveat=points,reltol=reltol,abstol=abstol,dt=dt,dtmin=dtmin,force_dtmin=true,maxiters=maxiter)

    ρ=fetch(ρ) ; σ=fetch(σ)

    M=zero(state.R)
    for i in 1:size(points)[1]
        t=points[i]
        J=sqrt(2)*jsource(f(t))
        w=weights[i]*f_prime(t)
        rho=ρ[i]
        sigma=σ[end-i+1]
        #order 0
        M+=w*state.ρ0*state.Rd*sigma[1,:,:]
        M+= - w*state.ρ0*sigma[1,:,:]*state.Rd
        #order 1
        M+=w*rho[1,:,:]*state.Rd*sigma[2,:,:] +w*J*(state.ρ0*sigma[2,:,:]*state.Q-state.Q*state.ρ0*sigma[2,:,:])
        M+=- w*rho[1,:,:]*sigma[2,:,:]*state.Rd
        M+=-w*J*(state.R*state.ρ0*sigma[2,:,:]-state.ρ0*sigma[2,:,:]*state.R)*state.Rd
        #order 2
        M+=w*rho[2,:,:]*state.Rd*sigma[3,:,:]
        M+=- w*rho[2,:,:]*sigma[3,:,:]*state.Rd
        #order3
        M+=w*rho[3,:,:]*state.Rd
        M+= w*J*(rho[2,:,:]*state.Q - state.Q*rho[2,:,:])
        M+= -w*rho[3,:,:]*state.Rd
        M+=- w*J*(state.R*rho[2,:,:]-rho[2,:,:]*state.R)*state.Rd
    end

    #grad=adjoint(robust_inv(state.ρ0,state.manifold_param.metric_reg)*M)
    grad=adjoint(robust_inv(state.ρ0,M,state.manifold_param.metric_reg))
    kin2=real(tr(ρ(tbound)[3,:,:]))

    return kin2,grad
end

function fg_phi4(state::State,g::Number)
    job1=@spawn kinetic1(state)
    job2=@spawn kinetic2(state)
    job3=@spawn phi_monomial(state,4)
    res1=fetch(job1)
    res2=fetch(job2)
    res3=fetch(job3)
    kin1=res1[1];∇kin1=res1[2]
    kin2=res2[1];∇kin2=res2[2]
    phi4=res3[1];∇phi4=res3[2]

    ∇energy=∇kin1+∇kin2+g*∇phi4
    energy=real(kin1+kin2+g*phi4)

    return energy,∇energy
end

function fg_sgordon(state::State,β::Number)
    job1=@spawn kinetic1(deepcopy(state))
    job2=@spawn kinetic2(deepcopy(state))
    job3=@spawn vertex(deepcopy(state),β)
    job4=@spawn vertex(deepcopy(state),-β)
    job5=@spawn phi_monomial(deepcopy(state),2)
    res1=fetch(job1) 
    res2=fetch(job2)
    res3=fetch(job3)
    res4=fetch(job4)
    res5=fetch(job5)
    kin1=res1[1];∇kin1=res1[2]
    kin2=res2[1];∇kin2=res2[2]
    vplus=res3[1];∇vplus=res3[2]
    vminus=res4[1];∇vminus=res4[2]
    mass=res5[1];∇mass=res5[2]

    ∇energy=∇kin1+∇kin2+0.5*1/β^2*(∇vplus+∇vminus) - 0.5*∇mass
    energy=real(kin1+kin2+0.5*1/β^2*(vplus+vminus) - 0.5*mass)

    return energy,∇energy
end

function Y(m1,m2)
    return 0.25/(2*π)*(m2^2- m1^2 + 2*m1^2*log(m1/m2))
end

function Z(m1,m2)
    return 1/(2*π)*log(m2/m1)
end

function fg_phi4_massmod(state::State,g::Number,massmod=.1)
    job1=@spawn kinetic1(state)
    job2=@spawn kinetic2(state)
    job3=@spawn phi_monomial(state,4)
    job4=@spawn phi_monomial(state,2)
    res1=fetch(job1)
    res2=fetch(job2)
    res3=fetch(job3)
    res4=fetch(job4)
    kin1=res1[1];∇kin1=res1[2]
    kin2=res2[1];∇kin2=res2[2]
    phi4=res3[1];∇phi4=res3[2]
    phi2=res4[1];∇phi2=res4[2]

    ∇energy=∇kin1+∇kin2 + (massmod^2-1)/2*∇phi2 + g*massmod^2*(∇phi4 - 6 * Z(massmod,1)* ∇phi2)
    energy= kin1 + kin2 + (massmod^2-1)/2*phi2 + Y(massmod,1) + g*massmod^2*(phi4 - 6 * Z(massmod,1)*phi2 +3 *Z(massmod,1)^2)
    GC.gc()
    return (1/massmod^2)*real(energy),(1/massmod^2)*∇energy
end

