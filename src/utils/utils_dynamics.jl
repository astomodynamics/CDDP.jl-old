
mutable struct ModelDimension <: AbstractDDPParameter
    nx::Int64 # state dimension
    nu::Int64 # control dimension
    nw::Int64 # noise dimension
    nλ::Int64 # constraint dimension
    ny::Int64 # observation dimension
    nv::Int64 # measurement noise dimension

    function ModelDimension(;nx=0, nu=0, nw=0, nλ=0, ny=0, nv=0)
        new(nx, nu, nw, nλ, ny, nv)
    end
end

struct DynamicsFunction <: AbstractDDPFunction
    f::Function # dynamic equation of motion without noise (out-of-place)
    f!::Function # dynamic equation of motion without noise (in-place)
    ∇f::Function # derivative of dynamics
    ∇²f::Function # second derivative of dynamics

    G::Function # noise matrix (out-of-place)
    G!::Function # noise matrix (in-place)
    ∇G::Function # derivative of noise matrix
    ∇²G::Function # second derivative of noise matrix

    cont_ode::Function # continuous ode function
    disc_ode::Function # discrete ode function

    function DynamicsFunction(;
        f=empty, 
        ∇f=empty, 
        ∇²f=empty, 
        f! =empty, 
        G=empty, 
        ∇G=empty, 
        ∇²G=empty,
        G! =empty, 
        integrator=:Tsit5(),
    )
        if isequal(f!, empty)
            cont_ode = f
        else
            cont_ode = f!
        end

        disc_ode(x, p, dt) = begin 
            prob = ODEProblem(cont_ode, x, (0.0, dt), p)
            X = solve(prob, integrator)
            return X[end]
        end
        
        new(
            f,
            f!,
            ∇f,
            ∇²f,
            G,
            G!,
            ∇G,
            ∇²G,
            cont_ode,
            disc_ode,
        )
    end
end

struct MPPIDynamicsFunction <: AbstractMPPIFunction
    f::Function # dynamic equation of motion without noise (out-of-place)
    f!::Function # dynamic equation of motion without noise (in-place)


    G::Function # noise matrix (out-of-place)
    G!::Function # noise matrix (in-place)

    function MPPIDynamicsFunction(;
        f=empty, 
        f! =empty, 
        G=empty, 
        G! =empty,
    )
        
        new(
            f,
            f!,
            G,
            G!,
        )
    end
end

function get_dims(model::AbstractDynamicsModel)
    if isequal(model.dims.nw, 0) && isequal(model.dims.nλ, 0)
        return model.dims.nx, model.dims.nu
    elseif isequal(model.dims.nw, 0)
        return model.dims.nx, model.dims.dim_u, model.dims.nλ
    elseif isequal(model.dims.nλ, 0)
        return model.dims.nx, model.dims.nu, model.dims.nw
    else
        return model.dims.nx, model.dims.nu, model.dims.nw, model.dims.nλ
    end
end


"""
struct ODE parameters
    
"""
mutable struct ODEParameter <: AbstractDDPParameter
    params::Any
    U_ref::Any
    U_md::Any
    X_ref::Any

    l::Any
    L::Any

    function ODEParameter(;
        params=nothing,
        U_ref=nothing,
        U_md=nothing,
        X_ref=nothing,
        l=nothing,
        L=nothing,
        )
        new(
            params,
            U_ref,
            U_md,
            X_ref,
            l,
            L,
        )
    end
end

"""

"""

# function get_ode_input(x, p, t)
#     δx = zeros(size(x,1))
#     U_ref = p.U_ref
#     X_ref = p.X_ref
#     u = nothing

#     # check if the reference control is array or function
#     if isa(U_ref, Vector)
#         u = U_ref 
#     else
#         # u = p.U_ref[trunc(Int, t/model.dt)+1]
#         u = U_ref(t)
#     end

#     # if the reference trajectory and feedback gains are given, DO feedback control
#     if !isequal(X_ref, nothing)
#         x_ref = X_ref(t)
#         δx = x - x_ref
#         u = Uref(t)  + p.L(t) * δx
#     end

#     return [p.params; u]
# end



"""
    rk4_step(model, f, t, x, u, h)

Returns one step of runge-kutta ode step with fixed time length

# Arguments


# Returns
- `ẋ`:

"""
function rk4_step(
    f::Function,
    x::Vector{Float64},
    p::ODEParameter,
    t::Float64,
    h::Float64,
)
    k1 = f(x, p, t+0.0)
    k2 = f(x + h / 2.0 * k1, p, t + h / 2.0)
    k3 = f(x + h / 2.0 * k2, p, t + h / 2.0)
    k4 = f(x + h * k3, p, t + h)
    return (k1 + 2 * k2 + 2 * k3 + k4)/6
end


function rk2_step(
    f::Function,
    x::Vector{Float64},
    p::ODEParameter,
    t::Float64,
    h::Float64,
)
    k1 = f(x, p, t+0.0)
    k2 = f(x + h * k1, p, t + h)
    return (k1 + k2)/2
end


function euler_step(
    f::Function,
    x::Vector{Float64},
    p::ODEParameter,
    t::Float64,
    h::Float64,
)
    return f(x, p, t)
end


function get_continuous_dynamics(
    f::Function,
    x::Vector{Float64},
    p::ODEParameter,
    t::Float64,
)
    return f(x, p, t)
    
end


function get_discrete_dynamics(
    f::Function,
    x::Vector{Float64},
    p::ODEParameter,
    t::Float64,
    h::Float64,
    method::Symbol=:rk4,
    isoutofplace::Bool=true,
)
    if method == :rk4
        return x+ rk4_step(f, x, p, t, h) * h
    elseif method == :rk2
        return x + rk2_step(f, x, p, t, h) * h
    elseif method == :euler
        return x + euler_step(f, x, p, t, h) * h
    end
end