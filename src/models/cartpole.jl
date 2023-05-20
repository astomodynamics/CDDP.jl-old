#################################################################################################
#=
    Model for cart pole dynamics
        from MIT's lecture on Underactuated Robotics
=#
#################################################################################################

using LinearAlgebra
using Distributions
using Random

export CartPole

# CartPole as a Julia class
struct CartPole <: AbstractDynamicsModel
    x_dim::Int64 # total state dimension
    u_dim::Int64 # total control input dimension

    # simulation setting
    tf::Float64 # final time 
    tN::Int64 # number of discretization steps
    dt::Float64 # discretization step-size


    # Boundary  conditions
    x_init::Vector{Float64}
    x_final::Vector{Float64}

    # function storage
    f!::Function # dynamic equation of motion without noise
    ∇f::Function # derivative of dynamics

    # dynamics constants
    mc::Float64
    mp::Float64
    l::Float64
    g::Float64
    
    function CartPole()
        x_dim = 4
        u_dim = 1
    
    
        x_init = [
            0.
            pi-0.6
            0.
            0.
        ]
    
        x_final = [
            0.
            π
            0.
            0.
        ]
        
        tf = 5.0
        tN = 100
        dt = tf/tN
        
        mc = 1.0
        mp = 0.2
        l = 0.5
        g = 9.81

    
        new(
            x_dim,
            u_dim,
            tf,
            tN,
            dt,
            x_init,
            x_final,
            f!,
            ∇f,
            mc,
            mp,
            l,
            g
        )
    end
end


"""
    f!(dx, x, p, t)

The dynamic equation of motion.

# Arguments
- `dx`: state derivative at a given time step
- `x`: state at a given time step
- `p`: parameter arguments
- `t`: time
"""
function f!(dx::Vector, x::Vector, p::ODEParams, t::Float64)
    model = p.model
    δx = zeros(size(x,1))
    if p.isarray
        u = p.U_ref 
    else
        u = p.U_ref(t)
    end

    # if the reference trajectory and feedback gains are given do feedback control
    if !isequal(p.X_ref, nothing)
        x_ref = p.X_ref(t)
        δx = x - x_ref
        u = p.Uref(t)  + p.L(t) * δx
    end

    """ edit begins >>>"""  
    mc = model.mc  # mass of the cart in kg (10)
    mp = model.mp   # mass of the pole (point mass at the end) in kg
    l = model.l   # length of the pole in m
    g = model.g  # gravity m/s^2

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

    #############################
    # θ = x[2]
    # θ̇ = x[4]
    # q̈ = [
    #     (u[1] + mp * sin(θ) * (l * θ̇^2 + g * cos(θ))) / (mc + mp * sin(θ)^2)
    #     (-u[1] * cos(θ) - mp * l * θ̇^2 * cos(θ) * sin(θ) - (mc + mp) * g * sin(θ)) / l / (mc + mp * sin(θ)^2)
    # ]
    #############################

    # if p.isarray
    #     # dx = [
    #     #     q̇
    #     #     q̈
    #     # ]
    #     return [
    #             q̇
    #             q̈
    #         ]
    # else
    #     dx[1:2] = q̇
    #     dx[3:4] = q̈
    #     # return dx
    # end
    dx[1:2] = q̇
    dx[3:4] = q̈

    """<<< edit ends """
    return dx
end



function ∇f(x::Vector{Float64}, u::Vector{Float64}, t::Float64)

    ∇ₓf = zeros(4,4)
    ∇ₓf += Diagonal([1. ; 1. ; 1. ; 1. ])
    
    ∇ₓf[1,3] += 1/20
    ∇ₓf[2,4] += 1/20
    ∇ₓf[3,1] += 0.0
    ∇ₓf[3,2] += - (50*x[4]^2*cos(x[2]) + 981*cos(x[2])^2 - 981*sin(x[2])^2)/(2000*(cos(x[2])^2 - 6)) - (cos(x[2])*sin(x[2])*(50*sin(x[2])*x[4]^2 + 500*u[1] + 981*cos(x[2])*sin(x[2])))/(1000*(cos(x[2])^2 - 6)^2)
    ∇ₓf[3,4] += -(x[4]*sin(x[2]))/(20*(cos(x[2])^2 - 6))
    ∇ₓf[4,2] += (2943*cos(x[2]) + 25*x[4]^2*cos(x[2])^2 - 25*x[4]^2*sin(x[2])^2 + 2500*u[1]*cos(x[2]) + 4905*cos(x[2])*sin(x[2]))/(2000*(cos(x[2])^2 - 6)) - (cos(x[2])*sin(x[2])*(50*x[4]^2*cos(x[2]) + 500*u[1] + 981*cos(x[2])*sin(x[2])))/(1000*(cos(x[2])^2 - 6)^2)
    ∇ₓf[4,4] += -(x[4]*cos(x[2])*sin(x[2]))/(10*(cos(x[2])^2 - 6))

    ∇ᵤf = zeros(4,1)
    ∇ᵤf[3,1] += -1/(4*(cos(x[2])^2 - 6))
    ∇ᵤf[4,1] += cos(x[2])/(2*(cos(x[2])^2 - 6))

    return ∇ₓf, ∇ᵤf
end