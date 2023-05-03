#################################################################################################
#=
    Model for unicycle car dynamics
=#
#################################################################################################

using LinearAlgebra
using Distributions
using Random
using ForwardDiff

export Unicycle

# Car2D as a Julia class
struct Unicycle <: AbstractDynamicsModel
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
    
    function Car2D()
        x_dim = 3
        u_dim = 2
    
    
        x_init = [
            0.0
            0.0
            0.0
        ]
    
        x_final = [
            3.0
            3.0 
            pi/2
        ]
        tf = 5.0
        tN = 100
        dt = tf/tN
    
    
        new(
            x_dim,
            u_dim,
            tf,
            tN,
            dt,
            x_init,
            x_final,
            f!,
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
    # necessary part begins =>
    model = p.model
    δx = zeros(size(x,1))
    if p.isarray
        u = p.Uref 
    else
        u = p.Uref(t)
    end

    if !isequal(p.Xref, nothing)
        xref = p.Xref(t)
        δx = x - xref
        u = p.Uref(t)  + p.L(t) * δx
    end
    # <= necessary part ends

    """ edit begins >>>"""

    if isequal(p.diff_ind, 0)    
        dx[1] = u[1] * sin(x[3])
        dx[2] = u[1] * cos(x[3])
        dx[3] = u[2]
        return dx
        
    elseif isequal(p.diff_ind, 1)
        return u[1] * sin(x[3])
    elseif isequal(p.diff_ind, 2)
        return u[1] * cos(x[3])
    elseif isequal(p.diff_ind, 3)
        return u[2]
    end    
    
    """<<< edit ends """

    
end