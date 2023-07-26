
function x_hessian(problem, x, u, t)
    n = length(x)
    out = ForwardDiff.jacobian(
        x -> ForwardDiff.jacobian(y -> problem.f(y, ODEParam(problem.model, u, isarray=true), t), x), x)
    return reshape(out, n, n, n)
end

function u_hessian(problem, x, u, t)
    n = length(x)
    m = length(u)
    out = ForwardDiff.jacobian(
    u -> ForwardDiff.jacobian(y -> problem.f(x, ODEParam(problem.model, y, isarray=true), t), u), u)
    return reshape(out, n, m, m)
end

function xu_hessian(problem, x, u, t)
    n = length(x)
    m = length(u)
    out = ForwardDiff.jacobian(
    z -> ForwardDiff.jacobian(y -> problem.f(z, ODEParam(problem.model, y, isarray=true), t), u), x)
    return reshape(out, n, n, m)
end


"""
    get_ode_derivatives(model, x, p, t, )

# NOTE: there might be a more efficient way to find hessian matrix
"""
function get_ode_derivatives(
    f,
    x::Vector{Float64},
    u::Vector{Float64},
    params_arr;
    u_md::Vector{Float64}=nothing,
    isilqr=true,
)   

    x_dim = length(x)
    u_dim = length(u)
    t = 0.0

    ∇ₓf = zeros(x_dim, x_dim)
    ∇ᵤf = zeros(x_dim, u_dim)
    if isilqr
        # ForwardDiff.jacobian!(∇ₓf, (dx,x) -> problem.f!(dx, x, ODEParams(problem.model, u, isarray=true), t), dx, x)
        # ForwardDiff.jacobian!(∇ᵤf, (dx,u) -> problem.f!(dx, x, ODEParams(problem.model, u, isarray=true), t), dx, u)
        # ∇ₓf = ForwardDiff.jacobian((dx,x) -> problem.f!(dx, x, ODEParams(problem.model, u, isarray=true), t), dx, x)
        # ∇ᵤf = ForwardDiff.jacobian((dx,u) -> problem.f!(dx, x, ODEParams(problem.model, u, isarray=true), t), dx, u)
        if isnothing(u_md)
            ∇ₓf = ForwardDiff.jacobian(x -> f(x, ODEParameter(params=params_arr, U_ref=u), t), x)
            ∇ᵤf = ForwardDiff.jacobian(u -> f(x,  ODEParameter(params=params_arr, U_ref=u), t), u)
        else
            ∇ₓf = ForwardDiff.jacobian(x -> f(x, ODEParameter(params=params_arr, U_ref=u, U_md=u_md), t), x)
            ∇ᵤf = ForwardDiff.jacobian(u -> f(x, ODEParameter(params=params_arr, U_ref=u, U_md=u_md), t), u)
        end
        
        return ∇ₓf, ∇ᵤf
    else
        ∇ₓf = ForwardDiff.jacobian(x -> f(x, ODEParameter(params=problem.model.params.arr, U_ref=u), t), x)
        ∇ᵤf = ForwardDiff.jacobian(u -> f(x, ODEParameter(params=problem.model.params.arr, U_ref=u), t), u)
        ∇ₓₓf = zeros(x_dim, x_dim, x_dim)
        ∇ₓᵤf = zeros(x_dim, x_dim, u_dim)
        ∇ᵤᵤf = zeros(x_dim, u_dim, u_dim)
        # ∇ₓₓf = x_hessian(problem, x, u, t)
        # ∇ₓᵤf = xu_hessian(problem, x, u, t)
        # ∇ᵤᵤf = u_hessian(problem, x, u, t)
        return ∇ₓf, ∇ᵤf, ∇ₓₓf, ∇ₓᵤf, ∇ᵤᵤf
    end 
end

"""
"""
function get_obs_derivative(
    h,
    x::Vector, 
    v::Vector
    )
    H = ForwardDiff.jacobian(x -> h(x,v), x)
    return H
end


"""
    get_instant_cost_derivatives()
"""
function get_instant_cost_derivatives(
    ell::Function,
    x::Vector{Float64},
    u::Vector{Float64},
    x_ref::Vector{Float64}
)  
    
    ∇ₓell = zeros(size(x,1)) 
    ∇ᵤell = zeros(size(u,1)) 
    ∇ₓₓell = zeros(size(x,1), size(x,1)) 
    ∇ₓᵤell = zeros(size(x,1), size(u,1)) 
    ∇ᵤᵤell = zeros(size(u,1), size(u,1)) 

    ∇ₓell = ForwardDiff.gradient!(∇ₓell, x -> ell(x, u, x_ref=x_ref), x)
    ∇ᵤell = ForwardDiff.gradient!(∇ᵤell, u -> ell(x, u, x_ref=x_ref), u)
    ∇ₓₓell = ForwardDiff.hessian!(∇ₓₓell, x -> ell(x, u, x_ref=x_ref), x)
    ∇ₓᵤell = ForwardDiff.jacobian!(∇ₓᵤell, (u -> ForwardDiff.gradient(x -> ell(x, u, x_ref=x_ref), x)), u)
    ∇ᵤᵤell = ForwardDiff.hessian!(∇ᵤᵤell, u -> ell(x, u, x_ref=x_ref), u)

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

    ∇ₓϕ = ForwardDiff.gradient!(∇ₓϕ, x -> phi(x,x_ref= x_final), x)
    ∇ₓₓϕ = ForwardDiff.hessian!(∇ₓₓϕ, x -> phi(x, x_ref=x_final), x)
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


function get_observation_derivatives(func, x)
    ∇h = ForwardDiff.jacobian(x -> func(x), x)
    return ∇h
end
