



using LinearAlgebra
using Distributions
using Random
using ForwardDiff
# using StaticArrays

export TwoBody

# Linearized Spcaecraft Relative Motion Dynamics as a Julia class
struct TwoBody <: AbstractDynamicsModel
    x_dim::Int64 # total state dimension
    u_dim::Int64 # total control input dimension

    # simulation setting
    tN::Int64 # number of discretization steps
    tf::Float64 # final time 
    dt::Float64 # discretization step-size


    # Boundary  conditions
    x_init::Vector{Float64}
    x_final::Vector{Float64}

    # function storage
    f!::Function # dynamic equation of motion without noise

    # dynamics parameters
    μ::Float64 # orbital rate
    T_max::Float64 # maximum thrust
    m_dot::Float64 # mass flow rate
end

function TwoBody()
    x_dim = 7
    u_dim = 3

    tN = 500
    tf = 8.162395951775
    dt = tf/tN

    x_init = [
        -0.6591305462686493
        0.729428037963236
        8.705661299781555e-5
        -0.761836764550642
        -0.66987402467608
        3.616373031922582e-5
        1.0
    ]
    x_final = [
        -1.6094289322431081
        -0.30103745631693807
        0.03330694897911128
        0.17799626332920876
        -0.7302074224564185
        -0.01967380084728356
        0.7150984745633979
    ]

    μ = 1.0
    T_max = 0.044968450404803254
    m_dot = 0.03592927296466571

    TwoBody(x_dim, u_dim, tN, tf, dt, x_init, x_final, f!, μ, T_max, m_dot)
end

function f!(dx,x,p,t)
    # necessary begin =>
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
    # <= necessary end
    μ = model.μ
    T_max = model.T_max
    m_dot = model.m_dot
    r = norm(x[1:3])
 
    dx[1:3] = x[4:6]
    dx[4:6] = -μ/r^3 * x[1:3] + T_max/x[7] * u
    dx[7] = -norm(u)*m_dot
        
    return dx
end