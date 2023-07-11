#################################################################################################
#=
    Model for cart pole dynamics
        from MIT's lecture on Underactuated Robotics
=#
#################################################################################################

using LinearAlgebra
using Distributions
using Random
using Symbolics

export CartPole

struct DynamicsParameter
    mc::Float64
    mp::Float64
    l::Float64
    g::Float64
    np::Int64 # number of parameters
    arr::Vector{Float64} # parameter array
    function DynamicsParameter(;
        mc=1.0,
        mp=0.2,
        l=0.5,
        g=9.81,
        np=4,
        arr=[mc, mp, l, g],
    )
        new(
            mc,
            mp,
            l,
            g,
            np,
            arr
        )
    end
end

# CartPole as a Julia class
mutable struct CartPole <: AbstractDynamicsModel
    dims::ModelDimension # dimension of the state, control, constraint, and noise

    # Boundary  conditions
    x_init::Vector{Float64}
    x_final::Vector{Float64}

    # function storage
    f::Function # dynamic equation of motion (out-of-place)
    f!::Function # dynamic equation of motion without noise (in place)
    ∇f::Function # derivative of dynamics
    ∇²f::Function # second derivative of dynamics
    
    G::Function # noise matrix (out-of-place)
    G!::Function # noise matrix (in-place)
    ∇G::Function # derivative of noise matrix
    ∇²G::Function # second derivative of noise matrix

    # dynamics constants
    params::DynamicsParameter
    
    function CartPole(;)

        dims = ModelDimension(nx=4, nu=1)
    
        x_init = [
            0.
            0.
            0.
            0.
        ]
    
        x_final = [
            0.
            π
            0.
            0.
        ]
        
        params = DynamicsParameter(
            mc=1.0,
            mp=0.01,
            l=1.0,
            g=9.81,
            np=4
        )

        """ DO NOT EDIT BELOW THIS LINE """ 
        # dynamic equation of motion
        @variables t du[1:dims.nx] u[1:dims.nx] p[1:params.np+dims.nu] real=true
        du = collect(du)
        symbolic_dynamics!(du, u, p, t)
        f_base! = build_function(du, u, p, t, target=Symbolics.CTarget(), expression=Val{false})
        
        f!(dx, x, params, t) = begin
            p = get_ode_input(x, params, t)
            f_base!(dx, x, p, t)
        end

        # derivative of dynamics
        symbolic_∇ₓf! = Symbolics.jacobian(du, u)
        symbolic_∇ᵤf! = Symbolics.jacobian(du, p[params.np+1:params.np+dims.nu])
        symbolic_∇ₓₓf! = Symbolics.jacobian(symbolic_∇ₓf!, u)
        symbolic_∇ₓᵤf! = Symbolics.jacobian(symbolic_∇ₓf!, p[params.np+1:params.np+dims.nu])
        symbolic_∇ᵤᵤf! = Symbolics.jacobian(symbolic_∇ᵤf!, p[params.np+1:params.np+dims.nu])
        ∇ₓf_base! = build_function(symbolic_∇ₓf!, u, p, t, target=Symbolics.CTarget(), expression=Val{false})
        ∇ᵤf_base! = build_function(symbolic_∇ᵤf!, u, p, t, target=Symbolics.CTarget(), expression=Val{false})
        ∇ₓₓf_base! = build_function(symbolic_∇ₓₓf!, u, p, t, target=Symbolics.CTarget(), expression=Val{false})
        ∇ₓᵤf_base! = build_function(symbolic_∇ₓᵤf!, u, p, t, target=Symbolics.CTarget(), expression=Val{false})
        ∇ᵤᵤf_base! = build_function(symbolic_∇ᵤᵤf!, u, p, t, target=Symbolics.CTarget(), expression=Val{false})
        
        ∇f(x, u, params) = begin
            t = 0.
            p = [params.arr; u]
            nx = size(x, 1)
            nu = size(u, 1)
            ∇ₓf = zeros(nx, nx)
            ∇ᵤf = zeros(nx, nu)
            ∇ₓf_base!(∇ₓf, x, p, t)
            ∇ᵤf_base!(∇ᵤf, x, p, t)
            return ∇ₓf, ∇ᵤf
        end

        ∇²f(x, u, params) = begin
            t = 0.
            p = [params.arr; u]
            nx = size(x, 1)
            nu = size(u, 1)
            ∇ₓₓf = zeros(nx, nx, nx)
            ∇ₓᵤf = zeros(nx, nx, nu)
            ∇ᵤᵤf = zeros(nx, nu, nu)
            ∇ₓₓf_base!(∇ₓₓf, x, p, t)
            ∇ₓᵤf_base!(∇ₓᵤf, x, p, t)
            ∇ᵤᵤf_base!(∇ᵤᵤf, x, p, t)
            return ∇ₓₓf, ∇ₓᵤf, ∇ᵤᵤf
        end

        new(
            dims,
            x_init,
            x_final,
            f,
            f!,
            ∇f,
            ∇²f,
            empty,
            empty,
            empty,
            empty,
            params
        )
    end
end

function symbolic_dynamics!(du, u, p, t)
    x, θ, ẋ, θ̇ = u
    mc, mp, l, g, ctrl = p
    s, c = sincos(θ)

    # inertia terms
    H = [
        mc + mp mp*l*c
        mp*l*c mp*l^2
    ]

    # coriolis and centrifugal terms
    C = [
        0. -mp*l*s*θ̇
        0. 0.
    ]

    # gravity terms
    G = [
        0.
        mp*g*l*s
    ]

    # control input terms
    B = [
        1.
        0.
    ]

    q̈ = H \ (- C*[ẋ; θ̇]- G + B*ctrl[1])

    du[1] = ẋ
    du[2] = θ̇
    du[3] = q̈[1]
    du[4] = q̈[2]
end

# function f(x, p, t)
#     dx = zeros(4)
#     f!(dx, x, p, t)
#     return dx
# end 

"""
    f(x, p, t)

The dynamic equation of motion.

# Arguments
- `x`: state at a given time step
- `p`: parameter arguments
- `t`: time
"""
function f(x::Vector, p::ODEParameter, t::Float64)
    p = get_ode_input(x, p, t) # DO NOT EDIT THIS LINE
    mc, mp, l, g, u = p

    q = x[1:2]
    q̇ = x[3:4]

    s, c = sincos(q[2])

    # inertia terms
    H = [
        mc + mp mp*l*c
        mp*l*c mp*l^2
    ]

    # coriolis and centrifugal terms
    C = [
        0. -mp*l*s*q̇[2]
        0. 0.
    ]

    # gravity terms
    G = [
        0.
        mp*g*l*s
    ]

    # control input terms
    B = [
        1.
        0.
    ]

    q̈ = H \ (- C*q̇ - G + B*u[1])

    return [
        q̇
        q̈
    ]

    """<<< edit ends """
    #############################
    # θ = x[2]
    # θ̇ = x[4]
    # q̈ = [
    #     (u[1] + mp * sin(θ) * (l * θ̇^2 + g * cos(θ))) / (mc + mp * sin(θ)^2)
    #     (-u[1] * cos(θ) - mp * l * θ̇^2 * cos(θ) * sin(θ) - (mc + mp) * g * sin(θ)) / l / (mc + mp * sin(θ)^2)
    # ]
    #############################
end




# function ∇f(x::Vector{Float64}, u::Vector{Float64}, dt::Float64)

#     ∇ₓf = zeros(4,4)
#     ∇ₓf += Diagonal([1. ; 1. ; 1. ; 1. ])
    
#     ∇ₓf[1,3] += 1/20
#     ∇ₓf[2,4] += 1/20
#     ∇ₓf[3,1] += 0.0
#     ∇ₓf[3,2] += - (50*x[4]^2*cos(x[2]) + 981*cos(x[2])^2 - 981*sin(x[2])^2)/(2000*(cos(x[2])^2 - 6)) - (cos(x[2])*sin(x[2])*(50*sin(x[2])*x[4]^2 + 500*u[1] + 981*cos(x[2])*sin(x[2])))/(1000*(cos(x[2])^2 - 6)^2)
#     ∇ₓf[3,4] += -(x[4]*sin(x[2]))/(20*(cos(x[2])^2 - 6))
#     ∇ₓf[4,2] += (2943*cos(x[2]) + 25*x[4]^2*cos(x[2])^2 - 25*x[4]^2*sin(x[2])^2 + 2500*u[1]*cos(x[2]) + 4905*cos(x[2])*sin(x[2]))/(2000*(cos(x[2])^2 - 6)) - (cos(x[2])*sin(x[2])*(50*x[4]^2*cos(x[2]) + 500*u[1] + 981*cos(x[2])*sin(x[2])))/(1000*(cos(x[2])^2 - 6)^2)
#     ∇ₓf[4,4] += -(x[4]*cos(x[2])*sin(x[2]))/(10*(cos(x[2])^2 - 6))

#     ∇ᵤf = zeros(4,1)
#     ∇ᵤf[3,1] += -1/(4*(cos(x[2])^2 - 6))
#     ∇ᵤf[4,1] += cos(x[2])/(2*(cos(x[2])^2 - 6))

#     return ∇ₓf, ∇ᵤf
# end

