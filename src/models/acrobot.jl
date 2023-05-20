#################################################################################################
#=
    Model for acrobot dynamics
        from MIT's lecture on Underactuated Robotics
=#
#################################################################################################

using LinearAlgebra
using Distributions
using Random

export Acrobot

# Acrobot as a Julia class
struct Acrobot <: AbstractDynamicsModel
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

    # dynamics constants
    m1::Float64 
    m2::Float64
    l1::Float64
    l2::Float64
    I1::Float64 
    I2::Float64 
    g::Float64 
    
    function CartPole()
        x_dim = 4
        u_dim = 1
    
    
        x_init = [
            0.
            π-0.6
            0.
            0.
        ]
    
        x_final = [
            π    
            0.
            0.
            0.
        ]
        
        tf = 5.0
        tN = 100
        dt = tf/tN
        
        m1 = 1.0
        m2 = 1.0
        l1 = 1.0
        l2 = 1.0
        I1 = 1.0
        I2 = 1.0
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
            m1,
            m2,
            l1,
            l2,
            I1,
            I2,
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
function f!(dx::Vector, x::Vector, p::Parameters, t::Float64)
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
    m1 = model.m1
    m2 = model.m2  
    l1 = model.l1
    l2 = model.l2
    I1 = model.I1
    I2 = model.I2
    g = model.g 

    q = x[1:2]
    q̇ = x[3:4]

    s1, c1 = sincos(q[1])
    s2, c2 = sincos(q[2])
    s12, c12 = sincos(q[1] + q[2])

    # inertia terms
    H = [
        I1 + I2 + m2 * l1^2 + 2 * m2 * l1*l2*c2  I2 + m2 * l1*l2*c2
        I2 + m2 * l1*l2*c2                                   I2
    ]
    
    # Coriolis and centrifugal terms
    C = [
        -2 * m2 * l1*l2*s2*q̇[2]  -m2 * l1*l2*s2*q̇[2]
        m2 * l1*l2*s2*q̇[1]       0
    ]
    
    # gravity terms
    G = [
        (m1 * l1 + m2 * l1)*g*s1 + m2 * l2 * g * s12
        m2 * l2 * g * s12
    ]

    # control input
    B = [
        0.
        1.
    ]

    q̈ = inv(H)* (- C*q̇ - G + B*u[1])


    if p.isarray
        dx = [
            q̇
            q̈
        ]
    else
        dx[1:2] = q̇
        dx[3:4] = q̈
    end

    """<<< edit ends """
    return dx
end