#################################################################################################
#=
    Model for Hill-Clohessy-Wiltshire 
=#
#################################################################################################

using LinearAlgebra
using Distributions
using Random
using Symbolics

export HCW

struct DynamicsParameter
    ω::Float64
    r_scale::Float64
    v_scale::Float64
    np::Int64 # number of parameters
    arr::Vector{Float64} # parameter array
    function DynamicsParameter(;
        μ=3.986004415e+5, # gravitational parameter of earth (km³/s²)
        a=6.8642335934215095e+3, # semimajor axis (km)
        r_scale=200,
        v_scale=1,
        np=3,
        arr=[r_scale, v_scale],
    )
        ω = sqrt(μ/a^3)
        arr = [ω, r_scale, v_scale]
        new(
            ω,
            r_scale,
            v_scale,
            np,
            arr
        )
    end
end


# Linearized Spcaecraft Relative Motion Dynamics as a Julia class
struct HCW <: AbstractDynamicsModel
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

    h::Function # observation function

    # dynamics constants
    params::DynamicsParameter
    
    function HCW()
        dims = ModelDimension(nx=6, nu=3, nw=6, nv=3)
        params = DynamicsParameter()
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

        f(x, params, t) = begin
            p = get_ode_input(x, params, t)
            dx = zeros(size(x,1))
            f_base!(dx, x, p, t)
            return dx
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
            h,
            params,
        )
    end
end

function symbolic_dynamics!(du, u, p, t)
    x, y, z, ẋ, ẏ, ż = u
    ω, r_scale, v_scale, ux, uy, uz = p

    x = x * r_scale
    y = y * r_scale
    z = z * r_scale
    ẋ = ẋ * v_scale
    ẏ = ẏ * v_scale
    ż = ż * v_scale
    
    du[1] = ẋ / r_scale
    du[2] = ẏ / r_scale
    du[3] = ż / r_scale
    du[4] = (3*ω^2*x + 2*ω*ẏ)/v_scale + ux
    du[5] = -2*ω*ẋ/v_scale + uy
    du[6] = -ω^2*ż/v_scale + uz
end


# """
#     f!(dx, x, p, t)

# The dynamic equation of motion.

# # Arguments
# - `x`: state at a given time step
# - `p`: parameter arguments
# - `t`: time
# """
# function f(x::Vector, p::ODEParams, t::Float64)
#     # necessary begin =>
#     model = p.model
#     δx = zeros(size(x,1))
#     if p.isarray
#         u = p.U_ref 
#     else
#         u = p.U_ref(t)
#     end

#     # if the reference trajectory and feedback gains are given do feedback control
#     if !isequal(p.X_ref, nothing)
#         x_ref = p.X_ref(t)
#         δx = x - x_ref
#         u = p.Uref(t)  + p.L(t) * δx
#     end
#     # <= necessary end

#     """ edit begin >>>"""
#     r_scale = model.r_scale
#     v_scale = model.v_scale
#     ω = model.ω

#     x = [
#         x[1:3] * r_scale
#         x[4:6] * v_scale
#     ]

#     dx = [
#         x[4] / r_scale
#         x[5] / r_scale
#         x[6] / r_scale
#         (3*ω^2*x[1] + 2*ω*x[5]) / v_scale + u[1]
#         -2*ω*x[4] / v_scale + u[2]
#         -ω^2*x[3] / v_scale + u[3]
#     ]
    
#     """<<< edit end """
#     return dx
# end

# function F(x::Vector, p::ODEParams, t::Float64)
#     model = p.model
#     std = sqrt(model.variance)

    
#     """ edit here >>>"""
#     dx = std * [
#         1.0
#         1.0
#         1.0
#         1.0
#         1.0
#         1.0
#     ]
#     """<<< edit end """
#     return dx
# end
function get_ode_input(x, p, t)
    U = p.U_ref
    X_ref = p.X_ref
    U_md = p.U_md
    u = nothing
    œ = nothing

    if isnothing(U)
        u = nothing
    elseif isa(U, Vector)
        # check if the reference control is array or function
        u = U
    else
        u = U(t)
    end

    return [p.params; u]
end


function h(x::Vector, v)
    y = [
        norm(x[1:3])
        atan(x[1]/x[2])
        atan(x[3]/norm(x[1:2]))
    ]
    return y
end