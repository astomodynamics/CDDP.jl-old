#################################################################################################
#=
    Model for simple pendulum dynamics
=#
#################################################################################################

using LinearAlgebra
using Distributions
using Random

export Pendulum

struct DynamicsParameter
    m::Float64
    l::Float64
    b::Float64
    g::Float64
    np::Int64 # number of parameters
    arr::Vector{Float64} # parameter array
    function DynamicsParameter(;
        m=1.0,
        l=0.5,
        b=0.01
        g=9.81,
        np=4,
        arr=[m, l, b, g],
    )
        new(
            mc,
            l,
            b,
            g,
            np,
            arr
        )
    end
end



# pendulum as a Julia class
struct Pendulum <: AbstractDynamicsModel
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
    
    
    function Pendulum()
        dims = ModelDimension(nx=2, nu=1)
    
    
        x_init = [
            0.
            0.
        ]
    
        x_final = [
            π
            0.
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
            params,
        )
    end
end


function symbolic_dynamics!(du, u, p, t)
    θ, θ̇ = u
    m, l, b, g, ctrl = p
    s, c = sincos(θ)

    # inertia terms
    H = m*l^2

    # coriolis and centrifugal terms
    C = b

    # gravity terms
    G = m*g*l*s

    # control input terms
    B = [
        1.
        0.
    ]

    θ̈ = H \ (- C*θ̇- G + B*ctrl[1])

    du[1] = θ̇
    du[2] = q̈
end






# function f!(dx::Vector, x::Vector, p::AbstractParameter, t::Float64)
    # # necessary part begins =>
    # model = p.model
    # δx = zeros(size(x,1))
    # if p.isarray
    #     u = p.U_ref 
    # else
    #     u = p.U_ref(t)
    # end

    # # if the reference trajectory and feedback gains are given do feedback control
    # if !isequal(p.X_ref, nothing)
    #     x_ref = p.X_ref(t)
    #     δx = x - x_ref
    #     u = p.Uref(t)  + p.L(t) * δx
    # end
    # # <= necessary part ends

    # """ edit begins >>>"""  
    # m = model.m  # mass of the pole in kg 
    # l = model.l   # length of the pole in m
    # b = model.b  # damping coefficient
    # g = model.g  # gravity m/s^2

    # q = x[1]
    # q̇ = x[2]

    # s, c = sincos(q[1])

    # # inertia terms
    # H = m*l^2
    # # gravity terms
    # G = m*g*l*s
    # # coriolis and centrifugal terms
    # C = b
    # # control input terms
    # B = 1.

    # q̈ = H \ (- G - C * q̇ + B * u[1])

    # dx[1] = q̇
    # dx[2] = q̈
    
    # """<<< edit ends """
    # return dx
# end

function ∇f()
    nothing
end


function G()
    nothing
end

function ∇G()
    nothing
end