#################################################################################################
#=
    Model for Hill-Clohessy-Wiltshire 
=#
#################################################################################################

using LinearAlgebra
using Distributions
using Random
using ForwardDiff
# using StaticArrays

export HCW

# Linearized Spcaecraft Relative Motion Dynamics as a Julia class
struct HCW <: AbstractDynamicsModel
    x_dim::Int64 # total state dimension
    u_dim::Int64 # total control input dimension

    # simulation setting
    tN::Int64 # number of discretization steps
    tf::Float64 # final time 
    dt::Float64 # discretization step-size

    r_scale::Float64 # scaling factor for position
    v_scale::Float64 # scaling factor for velocity

    # Boundary  conditions
    x_init::Vector{Float64}
    x_final::Vector{Float64}

    # problem constraints
    xMax::Float64
    xMin::Float64
    uMax::Float64
    uMin::Float64

    # function storage
    f::Function # dynamic equation of motion without noise
    F::Function # noise map function
    h::Function # observation function

    # dynamics parameters
    ω::Float64 # orbital rate

    # stochastic parameters
    variance::Float64
    distribution::Normal{Float64}
    
    function HCW()
        x_dim = 6
        u_dim = 3
    
        r_scale = 200
        v_scale = 1
    
        x_init = [
            -93.89268872140511 / r_scale
            68.20928216330306 / r_scale
            34.10464108165153 / r_scale
            0.037865035768176944 / v_scale
            0.2084906865487613 / v_scale
            0.10424534327438065 / v_scale
        ]
    
        x_final = [
            -37.59664132226163 / r_scale
            27.312455860666148 / r_scale
            13.656227930333074 / r_scale
            0.015161970413423813 / v_scale
            0.08348413138390476 / v_scale
            0.04174206569195238 / v_scale
        ]
        tf = 5000.0
        tN = 500
        dt = tf/tN
    
        xMax = 1e4
        xMin = -1e4
        uMax = 80e-6
        uMin = -80e-6
    
        μ = 3.986004415e+5 # gravitational parameter of earth (km³/s²)
        a = 6.8642335934215095e+3 # semimajor axis (km)
        ω = sqrt(μ/a^3)
    
        mean = 0.0
        variance = 1e-9
        std = sqrt(variance)
        distribution = Normal(mean, std)
    
        new(
            x_dim,
            u_dim,
            tN,
            tf,
            dt,
            r_scale,
            v_scale,
            x_init,
            x_final,
            xMax,
            xMin,
            uMax,
            uMin,
            f,
            F,
            h,
            ω,
            variance,
            distribution,
        )
    end
end


"""
    f!(dx, x, p, t)

The dynamic equation of motion.

# Arguments
- `x`: state at a given time step
- `p`: parameter arguments
- `t`: time
"""
function f(x::Vector, p::ODEParams, t::Float64)
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

    """ edit begin >>>"""
    r_scale = model.r_scale
    v_scale = model.v_scale
    ω = model.ω

    x = [
        x[1:3] * r_scale
        x[4:6] * v_scale
    ]

    dx = [
        x[4] / r_scale
        x[5] / r_scale
        x[6] / r_scale
        (3*ω^2*x[1] + 2*ω*x[5]) / v_scale + u[1]
        -2*ω*x[4] / v_scale + u[2]
        -ω^2*x[3] / v_scale + u[3]
    ]
    
    """<<< edit end """
    return dx
end

function F(x::Vector, p::ODEParams, t::Float64)
    model = p.model
    std = sqrt(model.variance)

    
    """ edit here >>>"""
    dx = std * [
        1.0
        1.0
        1.0
        1.0
        1.0
        1.0
    ]
    """<<< edit end """
    return dx
end



function h(x::Vector, t::Float64)
    y = [
        norm(x[1:3])
        atan(x[1]/x[2])
        atan(x[3]/norm(x[1:2]))
    ]
    return y
end