
"""
struct ODE parameters
    
"""
struct ODEParams <: Parameters
    model::AbstractDynamicsModel
    Uref::Any
    Xref::Any
    l::Any
    L::Any
    isarray::Bool
    diff_ind::Int64
end

function ODEParams(
    model, 
    Uref;
    Xref=nothing,
    l=nothing,
    L=nothing,
    isarray::Bool=false,
    diff_ind::Int64=0
    )
    ODEParams(
        model,
        Uref,
        Xref,
        l,
        L,
        isarray,
        diff_ind
    )
end

"""
    initialize_trajectory(model)

Initialize the state and control trajectory

# Arguments
- `model`: AbstractDynamicsModel
- `x_init`:
- ``:
- ``:
- ``:
- ``:
- ``:
- ``:
- ``:
- ``:

# Return
- `X`: state time history
- `U`: control time history
"""
function initialize_trajectory(
    model::AbstractDynamicsModel;
    x_init::Vector{Float64}=model.x_init, 
    tf::Float64=model.tf,
    tN::Int64=model.tN,
    f!::Function=model.f!,
    F!::Function=empty,
    ode_alg=Tsit5(),
    sde_alg=EM(),
    reltol=1e-8, 
    abstol=1e-8,
    randomize::Bool=false,
    isstochastic::Bool=false,
)
    dt = tf/tN
    # initialize U array
    if randomize
        U = 1e-9 * rand(Float64,(model.u_dim, tN - 1)) # use this if you want randomized initial trajectory
    else
        U = zeros(tN, model.u_dim)
    end

    U = Vector[U[t, :] for t in axes(U,1)]
    # convert U array into U_func as a continuous function
    U_func = linear_interpolation((collect(LinRange(0.0, tf, tN)),), U, extrapolation_bc = Line())

    if !isstochastic
        # integrate through DifferentialEquations.jl
        # set ODE parameters (for control and storaged trajectory)
        p = ODEParams(model, U_func)
        
        # define ODE problem
        prob = ODEProblem(f!, x_init, (0.0,tf), p)

        # solve ODE problem
        X = solve(prob, ode_alg, reltol=reltol, abstol=abstol)
    else
        # integrate through DifferentialEquations.jl
        # set ODE parameters (for control and storaged trajectory)
        p = ODEParams(model, U_func)

        # define SDE problem
        prob = SDEProblem(f!, F!, x_init, (0.0, tf), p, noise=WienerProcess(0.0, 0.0, 0.0))
        
        # solve SDE problem
        X = solve(prob, sde_alg, dt=dt)
    end
    return X, U_func
end



"""


simulate dynamics given initial condition and control sequence

# Arguments

"""
function simulate_trajectory(
    model::AbstractDynamicsModel,
    x_init::Vector{Float64}, 
    U,
    tf::Float64,
    dt::Float64;
    f!::Function=model.f!,
    F!::Function=empty,
    Xref=nothing,
    l=nothing,
    L=nothing,
    isfeedback::Bool=false,
    ode_alg=Tsit5(),
    sde_alg=EM(),
    reltol=1e-8, 
    abstol=1e-8,
    randomize::Bool=false,
    isstochastic::Bool=false,
)   

    if !isstochastic
        # integrate through DifferentialEquations.jl
        if !isfeedback
            p = ODEParams(model, U)
        else
            p = ODEParams(model, U, Xref=Xref, l=l, L=L)
        end

        # define ODE problem
        prob = ODEProblem(f!, x_init, (0.0,tf), p)

        # solve ODE problem
        X = solve(prob, ode_alg, reltol=reltol, abstol=abstol)
    else
        # integrate through DifferentialEquations.jl
        # set ODE parameters (for control and storaged trajectory)
        if !isfeedback
            p = ODEParams(model, U)
        else
            p = ODEParams(model, U, Xref=Xref, l=l, L=L)
        end

        # define SDE problem
        prob = SDEProblem(f!, F!, x_init, (0.0, tf), p, noise=WienerProcess(0.0, 0.0, 0.0))
        
        # solve SDE problem
        X = solve(prob, sde_alg, dt=dt)
    end

    return X
end


function f(f!::Function, x::Vector, p::Parameters, t::Float64)
    dx = f!(zeros(size(x,1)), x, p, t)
    return dx
end

"""
    get_ode_derivatives(model, x, p, t, )

# NOTE: there might be a more efficient way to find hessian matrix

# NOTE: there are mainly three ways to compute jacobian. The first one is the fastest
    1. ForwardDiff.jacobian!(fx, (y,x) -> f_x_reduc!(y,x,u, model), dx, model.x_init)
    2. ForwardDiff.jacobian((y,x) -> f_x_reduc!(y,x,u, model), dx, model.x_init)
    3. ForwardDiff.jacobian(x -> f_x_reduc(x, u, t, model), model.x_init)
"""
function get_ode_derivatives(
    problem::AbstractDDPProblem,
    x::Vector{Float64},
    u::Vector{Float64},
    t::Float64;
    isilqr=false,
)   

    x_dim, u_dim = problem.x_dim, problem.u_dim
    dx = zeros(x_dim)

    ∇ₓf = zeros(x_dim, x_dim)
    ∇ᵤf = zeros(x_dim, u_dim)
    if isilqr
        ForwardDiff.jacobian!(∇ₓf, (dx,x) -> problem.f!(dx, x, ODEParams(problem.model, u, isarray=true), t), dx, x)
        ForwardDiff.jacobian!(∇ᵤf, (dx,u) -> problem.f!(dx, x, ODEParams(problem.model, u, isarray=true), t), dx, u)

        return ∇ₓf, ∇ᵤf
    else
        ForwardDiff.jacobian!(∇ₓf, (dx,x) -> problem.f!(dx, x, ODEParams(problem.model, u, isarray=true), t), dx, x)
        ForwardDiff.jacobian!(∇ᵤf, (dx,u) -> problem.f!(dx, x, ODEParams(problem.model, u, isarray=true), t), dx, u)
        ∇ₓₓf = zeros(x_dim, x_dim, x_dim)
        ∇ₓᵤf = zeros(x_dim, x_dim, u_dim)
        ∇ᵤᵤf = zeros(x_dim, u_dim, u_dim)
        for i in 1:x_dim
            ForwardDiff.hessian!(
                ∇ₓₓf[i,:,:], x -> f(problem.f!, x, ODEParams(problem.model, u, isarray=true, diff_ind=i), t), x)
            ForwardDiff.hessian!(
                ∇ᵤᵤf[i,:,:], u -> f(problem.f!, x, ODEParams(problem.model, u, isarray=true, diff_ind=i), t), u)
            # ForwardDiff.jacobian!(
            #     ∇ₓᵤf[i,:,:], u -> ForwardDiff.gradient(
            #         x -> f(problem.f!, x, ODEParams(problem.model, u, isarray=true, diff_ind=i), t), x), u)
        end
        return ∇ₓf, ∇ᵤf, ∇ₓₓf, ∇ₓᵤf, ∇ᵤᵤf
    end 
end


"""
    get_instant_cost_derivatives()
"""
function get_instant_cost_derivatives(
    ell::Function,
    x::Vector{Float64},
    u::Vector{Float64},
    x_final::Vector{Float64}
)  
    
    ∇ₓell = zeros(size(x,1)) 
    ∇ᵤell = zeros(size(u,1)) 
    ∇ₓₓell = zeros(size(x,1), size(x,1)) 
    ∇ₓᵤell = zeros(size(x,1), size(u,1)) 
    ∇ᵤᵤell = zeros(size(u,1), size(u,1)) 

    ∇ₓell = ForwardDiff.gradient!(∇ₓell, x -> ell(x, u, x_final), x)
    ∇ᵤell = ForwardDiff.gradient!(∇ᵤell, u -> ell(x, u, x_final), u)
    ∇ₓₓell = ForwardDiff.hessian!(∇ₓₓell, x -> ell(x, u, x_final), x)
    ∇ₓᵤell = ForwardDiff.jacobian!(∇ₓᵤell, (u -> ForwardDiff.gradient(x -> ell(x, u, x_final), x)), u)
    ∇ᵤᵤell = ForwardDiff.hessian!(∇ᵤᵤell, u -> ell(x, u, x_final), u)

    return ∇ₓell, ∇ᵤell, ∇ₓₓell, ∇ₓᵤell, ∇ᵤᵤell
end


"""
    get_terminal_cost_derivatives()
"""
function get_terminal_cost_derivatives(
    phi::Function,
    x::Vector{Float64},
    x_final::Vector{Float64}
)
    ∇ₓϕ = zeros(size(x,1)) 
    ∇ₓₓϕ = zeros(size(x,1), size(x,1)) 

    ∇ₓϕ = ForwardDiff.gradient!(∇ₓϕ, x -> phi(x, x_final), x)
    ∇ₓₓϕ = ForwardDiff.hessian!(∇ₓₓϕ, x -> phi(x, x_final), x)
    return ∇ₓϕ, ∇ₓₓϕ
end


"""
"""
function get_instant_const_derivative(
    c::Function,
    x::Vector{Float64},
    u::Vector{Float64},
)
    x_dim = size(x,1)
    u_dim = size(u,1)
    λ_dim = size(c(zeros(x_dim), zeros(u_dim)), 1)

    ∇ₓc = zeros(λ_dim, x_dim) 
    ∇ᵤc = zeros(λ_dim, u_dim) 
    ∇ₓₓc = zeros(λ_dim, x_dim, x_dim) 
    ∇ₓᵤc = zeros(λ_dim, x_dim, u_dim) 
    ∇ᵤᵤc = zeros(λ_dim, u_dim, u_dim) 
    if isequal(λ_dim,1) 
        ∇ₓc = zeros(size(x,1)) 
        ∇ᵤc = zeros(size(u,1)) 
        ForwardDiff.gradient!(∇ₓc, x -> c(x, u), x)
        ForwardDiff.gradient!(∇ᵤc, u -> c(x, u), u)
    
        ForwardDiff.hessian!(∇ₓc, x -> c(x, u), x)
        ForwardDiff.hessian!(∇ᵤc, u -> c(x, u), u)
    else
        ForwardDiff.jacobian!(∇ₓc, x -> c(x, u), x)
        ForwardDiff.jacobian!(∇ᵤc, u -> c(x, u), u)
        for i in 1:λ_dim
            ForwardDiff.hessian!(∇ₓₓc[i,:,:], x -> c(x, u)[i], x)
            ForwardDiff.hessian!(∇ᵤᵤc[i,:,:], u -> c(x, u)[i], u)
        end
    end

    return ∇ₓc, ∇ᵤc, ∇ₓₓc, ∇ₓᵤc, ∇ᵤᵤc
end

"""
"""
function get_terminal_const_derivative(
    c_final::Function,
    x::Vector{Float64},
    u::Vector{Float64},
)
   
    x_dim = size(x,1)
    u_dim = size(u,1)
    λ_dim = size(c_final(zeros(x_dim), zeros(u_dim)), 1)

    ∇ₓc = zeros(λ_dim, x_dim) 
    ∇ᵤc = zeros(λ_dim, u_dim) 
    ∇ₓₓc = zeros(λ_dim, x_dim, x_dim) 
    ∇ₓᵤc = zeros(λ_dim, x_dim, u_dim) 
    ∇ᵤᵤc = zeros(λ_dim, u_dim, u_dim) 

    if isequal(λ_dim,1) 
        ∇ₓc = zeros(size(x,1)) 
        ∇ᵤc = zeros(size(u,1)) 
        ForwardDiff.gradient!(∇ₓc, x -> c(x, u), x)
        ForwardDiff.gradient!(∇ᵤc, u -> c(x, u), u)


    else
        ForwardDiff.jacobian!(∇ₓc, x -> c(x, u), x)
        ForwardDiff.jacobian!(∇ᵤc, u -> c(x, u), u)
    end

    return ∇ₓc, ∇ᵤc, ∇ₓₓc, ∇ₓᵤc, ∇ᵤᵤc
end


"""
    rk4_step(model, f, t, x, u, h)

Returns one step of runge-kutta ode step with fixed time length

# Arguments


# Returns
- `ẋ`:

"""
function rk4_step(
    f!::Function,
    x::Vector{Float64},
    p::Parameters,
    t::Float64;
    h::Float64=model.dt,
)
    dx = zeros(size(x,1))
    k1 = f!(dx, x, p, t+0.0)
    k2 = f!(dx, x + h / 2.0 * k1, p, t + h / 2.0)
    k3 = f!(dx, x + h / 2.0 * k2, p, t + h / 2.0)
    k4 = f!(dx, x + h * k3, p, t + h)
    return (k1 + 2 * k2 + 2 * k3 + k4)/6
end


function rk2_step(
    f!::Function,
    x::Vector{Float64},
    p::Parameters,
    t::Float64;
    h::Float64=model.dt,
)
    dx = zeros(size(x,1))
    k1 = f!(dx, x, p, t+0.0)
    k2 = f!(dx, x + h * k1, p, t + h)
    return (k1 + k2)/2
end


function euler_step(
    f!::Function,
    x::Vector{Float64},
    p::Parameters,
    t::Float64;
    h::Float64=model.dt,
)
    return f!(zeros(size(x,1)), x, p, t)
end


"""
    get_feasibility(model, X, U)
Check feasibility
"""
function get_feasibility(
    prob::AbstractDDPProblem,
    X,
    U,
)
    dt = prob.dt
    for k in 0:prob.tN-1
        c = prob.c(X(k*dt), U(k*dt))
        for c_ele in c
            if c_ele >= 0
                return false
            end
        end
    end
    return true
end


"""
"""
function get_trajectory_cost(
    X,
    U,
    X_ref,
    x_final,
    ell::Function,
    ϕ::Function,
    tN::Int64,
    dt::Float64,
)
    J = 0
    for k in 1:tN
        if isequal(X_ref, nothing)
            J += ell(X(k*dt), U(k*dt), zeros(axes(X(k*dt), 1)))
        else
            J += ell(X(k*dt), U(k*dt), X_ref(k*dt))
        end
        # J += ell(X(k*dt), U(k*dt), x_final) * dt
    end
    
    J += ϕ(X(tN*dt), x_final)
    return J
end


function get_trajectory_log_cost(
    prob::AbstractDDPProblem,
    params::CDDPParameter,
    X,
    U,
    Y,
    isfeasible,
)
    L = get_trajectory_cost(X, U, prob.X_ref, prob.x_final, prob.ell, prob.ϕ, prob.tN, prob.dt) 
    for k in 0:prob.tN-1
        if isfeasible

            c = prob.c(X(k*prob.dt), U(k*prob.dt))
            for i in axes(c,1)
                L -= params.μip * log(-c[i])
            end
        else

            y = Y(k*prob.dt)
            for i in axes(y,1)
                L -= params.μip * log(y[i])
            end
        end
    end
    return L
end
