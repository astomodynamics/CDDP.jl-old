#################################################################################################
#=
    Model for attitude dynamics using quaternion
=#
#################################################################################################


# using ForwardDiff
using Distributions
using Random
using ForwardDiff

# Dynamics model
abstract type AbstractModel end

export Quaternion

# Linearized Spcaecraft Relative Motion Dynamics as a Julia class
mutable struct Quaternion <: AbstractModel
    x_dim::Int64 # total state dimension
    u_dim::Int64 # total control input dimension
    λ_dim::Int64 # constraints dimension

    # Boundary  conditions
    x_init::AbstractArray{Float64,1}
    x_final::AbstractArray{Float64,1}

    # simulation setting
    dt::Float64 # discretization step-size
    tN::Int64 # number of time-discretization steps
    hN::Int64 # number of horizon time-discretization steps

    # problem constraints
    xMax::Float64
    xMin::Float64
    uMax::Float64
    uMin::Float64

    dynamics::Function # dynamic equation of motion
    ode_step::Function

    # dynamics constants
    # ω::Float64 # orbital rate
    Inertia::AbstractArray{Float64,2}

    # weight matrices for terminal and running costs
    F::AbstractArray{Float64,2}  # terminal cost weight matrix
    Q::AbstractArray{Float64,2}  # state running cost weight matrix
    R::AbstractArray{Float64,2}  # control running cost weight matrix

    variance::Float64
    distribution::Normal{Float64}

    conv_tol::Float64 # convergence tolerance
    max_ite::Int64 # maximum iteration threshhold

    islinear::Bool
    isconstrained::Bool
    isstochastic::Bool
end


# Define the model struct with parameters
function Quaternion()
    x_dim = 7
    u_dim = 3
    λ_dim = 0

    ω_init = [
        0.0001
        -0.0001
        0.0001
    ]
    q_init = [
        sqrt(0.32)
        sqrt(0.32)
        sqrt(0.32)
        -0.2
    ]
    x_init = [
        q_init
        ω_init
    ]


    ω_final = [
        0
        0
        0
    ]
    q_final = [
        0.
        0.
        0.
        -1
    ]
    x_final = [
        q_final
        ω_final
    ]

    dt = 10
    tN = 500
    hN = 100

    xMax = 1e3
    xMin = -1e3
    uMax = 80e-2
    uMin = -80e-2

    ode_step = rk4_step

    F = Diagonal([1e+2 * [1; 1; 1; 1]; 1e+2 * [1; 1; 1]])
    Q = zeros(x_dim, x_dim) # Diagonal([1e-0 * [1; 1]; 1e-0 * [1; 1]])
    R = Diagonal(1e+2*[1; 1; 1])

    Inertia = [
        4 0 0
        0 2 0 
        0 0 4
    ]

    mean = 0.0
    variance = 1e+0
    deviation = sqrt(variance)
    distribution = Normal(mean, deviation)
    conv_tol = 1e-5
    max_ite = 10

    islinear = true
    isconstrained = []
    if λ_dim == 0
        isconstrained = false
    else
        isconstrained = true
    end
    isstochastic = false

    Quaternion(
        x_dim,
        u_dim,
        λ_dim,
        x_init,
        x_final,
        dt,
        tN,
        hN,
        xMax,
        xMin,
        uMax,
        uMin,
        dynamics,
        ode_step,
        Inertia,
        F,
        Q,
        R,
        variance,
        distribution,
        conv_tol,
        max_ite,
        islinear,
        isconstrained,
        isstochastic,
    )
end


"""
    dynamics(model, x, u, t, step)

The dynamic equation of motion.

# Arguments
- `model`: Abstract model
- `x`: state at a given time step
- `u`: control at a given time step
- `t`: time at a given time step
- `step`: time step

# Returns
- `ẋ`: time derivative of nonlinear equation of motion
"""
function dynamics(model::Quaternion, x::Vector, u::Vector, t::Float64, step::Int64)
    q = x[1:4] # extract quaternion
    ω = x[5:7] # extract angular velocity vector
    J = model.Inertia # extract inertia tensor

    # compute skew matrix
    q_skew = get_skew_matrix(q[1:3])
    Ξ = [
        q[4] * I  + q_skew
        -q[1:3]'
    ]

    # quaternion kinematics 
    q̇ = 1/2 * Ξ * ω

    # angular momentum dynamics
    ω_skew = get_skew_matrix(ω)
    ω̇ = -J^-1 * (ω_skew * J * ω - u)

    ẋ = [
        q̇
        ω̇
    ]
    return ẋ
end


function get_skew_matrix(x::Vector)
    return [
        0 -x[3] x[2]
        x[3] 0 -x[1]
        -x[2] x[1] 0
    ]
end


"""
    get_dynamics_jac_hess()
"""
function get_dynamics_jac_hess(
    model::Quaternion,
    x::AbstractArray{Float64,1},
    u::AbstractArray{Float64,1},
    t::Int64;
    islinear::Bool=model.islinear,
    isstochastic::Bool=model.isstochastic,
)
    x_dim, u_dim = model.x_dim, model.u_dim
    fx = zeros(x_dim, x_dim)
    fu = zeros(x_dim, u_dim)
    fxx = zeros(x_dim, x_dim, x_dim)
    fxu = zeros(x_dim, x_dim, u_dim)
    fuu = zeros(x_dim, u_dim, u_dim)
    gx = zeros(x_dim, x_dim)
    gu = zeros(x_dim, u_dim)
    gxx = zeros(x_dim, x_dim, x_dim)
    gxu = zeros(x_dim, x_dim, u_dim)
    guu = zeros(x_dim, u_dim, u_dim)

    # # dynamics_funcs = get_dynamics_funcs(model)
    # dynamics = model.dynamics

    fx = ForwardDiff.jacobian(x -> model.dynamics(model, x, u, 0.0, t), x)
    fu = ForwardDiff.jacobian(u -> model.dynamics(model, x, u, 0.0, t), u)

    # ω = model.ω
    # fx = [
    #     0. 0. 0. 1. 0. 0
    #     0. 0. 0. 0. 1. 0
    #     0. 0. 0. 0. 0. 1.
    #     3*ω^2 0. 0. 0. 2*ω 0.
    #     0. 0. 0. -2*ω 0. 0.
    #     0. 0. -ω^2 0. 0. 0.
    # ]
    
    # fu = [
    #     0. 0. 0.
    #     0. 0. 0.
    #     0. 0. 0.
    #     1. 0. 0.
    #     0. 1. 0.
    #     0. 0. 1.
    # ]

    return fx, fu, fxx, fxu, fuu, gx, gu, gxx, gxu, guu
end