#################################################################################################
#=
    Model for inverted pendulum dynamics
=#
#################################################################################################

using LinearAlgebra
using Distributions
using Random

export Pendulum

# CartPole as a Julia class
struct Pendulum <: AbstractDynamicsModel
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
    ∇f::Function # Jacobian of the dynamic equation of motion
    G:: Function # noise matrix
    ∇G::Function # Jacobian of the noise matrix

    # dynamics constants
    m::Float64
    l::Float64
    b::Float64
    g::Float64
    
    
    function Pendulum()
        x_dim = 2
        u_dim = 1
    
    
        x_init = [
            0.
            0.
        ]
    
        x_final = [
            π
            0.
        ]
        
        tf = 5.0
        tN = 100
        dt = tf/tN
        
        m = 1.0
        l = 1.0
        b = 0.1
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
            m,
            l,
            b,
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
function f!(dx::Vector, x::Vector, p::AbstractParameter, t::Float64)
    # necessary part begins =>
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
    # <= necessary part ends

    """ edit begins >>>"""  
    m = model.m  # mass of the pole in kg 
    l = model.l   # length of the pole in m
    b = model.b  # damping coefficient
    g = model.g  # gravity m/s^2

    q = x[1]
    q̇ = x[2]

    s, c = sincos(q[1])

    # inertia terms
    H = m*l^2
    # gravity terms
    G = m*g*l*s
    # coriolis and centrifugal terms
    C = b
    # control input terms
    B = 1.

    q̈ = H \ (- G - C * q̇ + B * u[1])

    dx[1] = q̇
    dx[2] = q̈
    
    """<<< edit ends """
    return dx
end

function ∇f()
    nothing
end


function G()
    nothing
end

function ∇G()
    nothing
end