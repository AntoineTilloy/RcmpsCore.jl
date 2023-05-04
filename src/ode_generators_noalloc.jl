using SpecialFunctions 
using OrdinaryDiffEq
import Octavian: matmul, matmul! 




#Source term in the ODE
function jsource(y::Real)
    return besselk(0.25,abs(y))/(2^(9/4)*sqrt(pi)*gamma(1.25)*abs(y)^(1/4))
end

#Functions for the change of time variable in the ODE integration
function f(t::Real)
    return exp(-t^2)*t^3+(1-exp(-t^2))*t
end

function f_prime(t::Real)
    ex=exp(-t^2)
    return 3*t^2*ex - 2*t^4*ex +(1-ex) + 2*t^2*ex
end

function lindblad(state::State,ρ::AbstractMatrix)
    return state.Q*ρ +ρ*state.Qd + state.R*ρ*state.Rd
end

function lindblad_adjoint(state::State,ρ::AbstractMatrix)
    return state.Qd*ρ + ρ*state.Q + state.Rd*ρ*state.R
end

function lindblad!(dρ::AbstractMatrix,state::State,ρ::AbstractMatrix)
    matmul!(dρ,state.Q,ρ)
    matmul!(dρ,ρ,state.Qd,1.0,1.0)
    matmul!(state.buffer1,ρ,state.Rd)
    matmul!(dρ,state.R,state.buffer1,1.0,1.0)
end

function lindblad_MKL!(dρ::AbstractMatrix,state::State,ρ::AbstractMatrix)
    id = one(eltype(state.buffer1))
    mul!(dρ,state.Q,ρ)
    mul!(dρ,ρ,state.Qd,id,id)
    mul!(state.buffer1,ρ,state.Rd)
    mul!(dρ,state.R,state.buffer1,id,id)
end



function lindblad_adjoint!(dρ::AbstractMatrix,state::State,ρ::AbstractMatrix)
    matmul!(dρ,state.Qd,ρ)
    matmul!(dρ,ρ,state.Q,1.0,1.0)
    matmul!(state.buffer1,ρ,state.R)
    matmul!(dρ,state.Rd,state.buffer1,1.0,1.0)
end

#Vertex operator

function vertex_generator!(dρ::AbstractMatrix,ρ::AbstractMatrix,p,t)
    state=p[1]
    β=p[2]
    lindblad!(dρ,state,ρ)
    matmul!(dρ,state.R,ρ,eltype(ρ)(β*f_prime(t)*jsource(f(t))),f_prime(t))
    matmul!(dρ,ρ,state.Rd,eltype(ρ)(β*f_prime(t)*jsource(f(t))),one(eltype(ρ)))
end 

function vertex_generator_adjoint!(dρ::AbstractMatrix,ρ::AbstractMatrix,p,t)
    state=p[1]
    β=p[2]
    lindblad_adjoint!(dρ,state,ρ)
    matmul!(dρ,state.Rd,ρ,eltype(ρ)(β*f_prime(t)*jsource(f(t))),f_prime(t))
    matmul!(dρ,ρ,state.R,eltype(ρ)(β*f_prime(t)*jsource(f(t))),one(eltype(ρ)))
end 

#Phi monomial
function phi_generator!(dρ::AbstractArray,ρ::AbstractArray,p,t)
    J=jsource(f(t))
    state=p[1]
    n=p[2]
    dρ[1,:,:] .= 0.0
    for i in 2:(n+1)
        lindblad!(state.buffer2,state,view(ρ,i,:,:))
        matmul!(state.buffer2,state.R,view(ρ,i-1,:,:),(i-1)*J*f_prime(t),f_prime(t))
        matmul!(state.buffer2,view(ρ,i-1,:,:),state.Rd,(i-1)*J*f_prime(t),1.0)
        dρ[i,:,:] = state.buffer2
        #dρ[i,:,:] = f_prime(t)*((i-1)* J * (state.R*ρ[i-1,:,:] + ρ[i-1,:,:] * state.Rd) + lindblad(state,ρ[i,:,:]))
    end
end

function phi_generator_adjoint!(dρ::AbstractArray,ρ::AbstractArray,p,t)
    J=jsource(f(t))
    state=p[1]
    n=p[2]
    dρ[n+1,:,:] .= 0.0
    for i in 1:n
        lindblad_adjoint!(state.buffer2,state,view(ρ,i,:,:))
        matmul!(state.buffer2,state.Rd,view(ρ,i+1,:,:),i*J*f_prime(t),f_prime(t))
        matmul!(state.buffer2, view(ρ,i+1,:,:),state.R,i*J*f_prime(t),1.0)
        dρ[i,:,:] = state.buffer2
        #dρ[i,:,:] = f_prime(t)*(i*J * (state.Rd*ρ[i+1,:,:] + ρ[i+1,:,:] * state.R) + lindblad_adjoint(state,ρ[i,:,:]))
    end
end

#First part of the kinetic term
function kinetic1_generator!(dρ::AbstractArray,ρ::AbstractArray,state,t)
    J=jsource(f(t))*sqrt(2)

    lindblad!(state.buffer2,state,view(ρ,1,:,:))
    matmul!(state.buffer2,state.R,state.ρ0,J*f_prime(t),f_prime(t))
    dρ[1,:,:] = state.buffer2 

    lindblad!(state.buffer2,state,view(ρ,2,:,:))
    matmul!(state.buffer2,state.ρ0,state.Rd,J*f_prime(t),f_prime(t))
    dρ[2,:,:] = state.buffer2 

    lindblad!(state.buffer2,state,view(ρ,3,:,:))
    matmul!(state.buffer2,state.R,view(ρ,2,:,:),J*f_prime(t),f_prime(t))
    matmul!(state.buffer2,view(ρ,1,:,:),state.Rd,J*f_prime(t),1.0)
    dρ[3,:,:] = state.buffer2 

    #dρ[1,:,:]=f_prime(t)*(lindblad(state,ρ[1,:,:]) + J*state.R*state.ρ0)
    #dρ[2,:,:]=f_prime(t)*(lindblad(state,ρ[2,:,:]) + J*state.ρ0*state.Rd)
    #dρ[3,:,:]=f_prime(t)*(lindblad(state,ρ[3,:,:]) + J*(state.R*ρ[2,:,:] + ρ[1,:,:]*state.Rd))
end

function kinetic1_generator_adjoint!(dρ::AbstractArray,ρ::AbstractArray,state,t)
    J=jsource(f(t))*sqrt(2)

    lindblad_adjoint!(state.buffer2,state,view(ρ,1,:,:))
    matmul!(state.buffer2,view(ρ,2,:,:),state.R,J*f_prime(t),f_prime(t))
    matmul!(state.buffer2,state.Rd,view(ρ,3,:,:),J*f_prime(t),1.0)
    dρ[1,:,:] = state.buffer2 

    lindblad_adjoint!(state.buffer2,state,view(ρ,2,:,:))
    matmul!(state.buffer2,state.Rd,one(state.ρ0),J*f_prime(t),f_prime(t))
    dρ[2,:,:] = state.buffer2 

    lindblad_adjoint!(state.buffer2,state,view(ρ,3,:,:))
    matmul!(state.buffer2,one(state.ρ0),state.R,J*f_prime(t),f_prime(t))
    dρ[3,:,:] = state.buffer2 

    # dρ[1,:,:]=f_prime(t)*(lindblad_adjoint(state,ρ[1,:,:]) + J*(ρ[2,:,:]*state.R+state.Rd*ρ[3,:,:]))
    # dρ[2,:,:]=f_prime(t)*(lindblad_adjoint(state,ρ[2,:,:]) + J*state.Rd)
    # dρ[3,:,:]=f_prime(t)*(lindblad_adjoint(state,ρ[3,:,:]) + J*state.R)

end

#Second part of the kinetic term
function kinetic2_generator!(dρ::AbstractArray,ρ::AbstractArray,state,t)
    J=jsource(f(t))*sqrt(2)

    lindblad!(state.buffer2,state,view(ρ,1,:,:))
    matmul!(state.buffer2,state.QRcomm,state.ρ0,J*f_prime(t),f_prime(t))
    dρ[1,:,:] = state.buffer2 

    lindblad!(state.buffer2,state,view(ρ,2,:,:))
    matmul!(state.buffer2,state.ρ0,state.RdQdcomm,J*f_prime(t),f_prime(t))
    dρ[2,:,:] = state.buffer2 

    lindblad!(state.buffer2,state,view(ρ,3,:,:))
    matmul!(state.buffer2,state.QRcomm,view(ρ,2,:,:),J*f_prime(t),f_prime(t))
    matmul!(state.buffer2,view(ρ,1,:,:),state.RdQdcomm,J*f_prime(t),1.0)
    dρ[3,:,:] = state.buffer2 

    # dρ[1,:,:]=f_prime(t)*(lindblad(state,ρ[1,:,:]) + J*state.QRcomm*state.ρ0)
    # dρ[2,:,:]=f_prime(t)*(lindblad(state,ρ[2,:,:]) + J*state.ρ0*state.RdQdcomm)
    # dρ[3,:,:]=f_prime(t)*(lindblad(state,ρ[3,:,:]) + J*(state.QRcomm*ρ[2,:,:]+ ρ[1,:,:]*state.RdQdcomm))
end

function kinetic2_generator_adjoint!(dρ::AbstractArray,ρ::AbstractArray,state,t)
    J=jsource(f(t))*sqrt(2)

    lindblad_adjoint!(state.buffer2,state,view(ρ,1,:,:))
    matmul!(state.buffer2,view(ρ,2,:,:),state.QRcomm,J*f_prime(t),f_prime(t))
    matmul!(state.buffer2,state.RdQdcomm,view(ρ,3,:,:),J*f_prime(t),1.0)
    dρ[1,:,:] = state.buffer2 

    lindblad_adjoint!(state.buffer2,state,view(ρ,2,:,:))
    matmul!(state.buffer2,state.RdQdcomm,one(state.ρ0),J*f_prime(t),f_prime(t))
    dρ[2,:,:] = state.buffer2 

    lindblad_adjoint!(state.buffer2,state,view(ρ,3,:,:))
    matmul!(state.buffer2,one(state.ρ0),state.QRcomm,J*f_prime(t),f_prime(t))
    dρ[3,:,:] = state.buffer2 

    # dρ[1,:,:]=f_prime(t)*(lindblad_adjoint(state,ρ[1,:,:]) + J*(ρ[2,:,:]*state.QRcomm + state.RdQdcomm*ρ[3,:,:]))
    # dρ[2,:,:]=f_prime(t)*(lindblad_adjoint(state,ρ[2,:,:]) + J*state.RdQdcomm)
    # dρ[3,:,:]=f_prime(t)*(lindblad_adjoint(state,ρ[3,:,:]) + J*state.QRcomm)
end


function g(t::Real,x)
    return -0.25*x/(π)*sin(4*π*t/x) + t
end

function g_prime(t::Real,x)
    return 1-cos(4*π*t/x)
end

function double_vertex_generator!(dρ::AbstractMatrix,ρ::AbstractMatrix,p,t)
    state=p[1]
    β1=p[2]
    β2=p[3]
    x=p[4]
    tau=g(t,x)
    J=β1*jsource(tau-x/2)+ β2*jsource(tau+x/2)
    dρ[:,:] = g_prime(t,x)*(lindblad(state,ρ) + J*(matmul(state.R,ρ) + matmul(ρ,state.Rd)))
end