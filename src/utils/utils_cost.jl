
struct CostFunction <: AbstractDDPFunction
    ell::Function
    ∇ell::Function
    ∇²ell::Function
    ϕ::Function
    ∇ϕ::Function
    ∇²ϕ::Function

    function CostFunction(;
        ell=empty,
        ∇ₓell=empty,
        ∇ᵤell=empty,
        ∇ₓₓell=empty,
        ∇ₓᵤell=empty,
        ∇ᵤᵤell=empty,
        phi=empty,
        ∇ₓphi=empty,
        ∇ₓₓphi=empty,
    )
        if isequal(∇ₓell,empty)
            ∇ell = empty
        else
            ∇ell(x::Vector, u::Vector; x_ref::Vector=nothing) = begin
                return [∇ₓell(x, u, x_ref=x_ref), ∇ᵤell(x, u, x_ref=x_ref)]
            end

            ∇²ell(x::Vector, u::Vector; x_ref::Vector=nothing) = begin
                return ∇ₓₓell(x, u,  x_ref=x_ref), ∇ₓᵤell(x, u,  x_ref=x_ref), ∇ᵤᵤell(x, u,  x_ref=x_ref)
            end
        end
        
        new(
            ell,
            ∇ell,
            ∇²ell,
            phi,
            ∇ₓphi,
            ∇ₓₓphi,
        )
    end
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
    for k in 0:tN-1
        t = k*dt
        if isequal(X_ref, nothing)
            J += ell(X(t), U(t), x_ref=zeros(axes(X(t), 1))) * dt
        else
            J += ell(X(t), U(t), x_ref=X_ref(t)) * dt
        end
    end
    
    J += ϕ(X(tN*dt), x_ref=x_final)
    return J
end


function get_trajectory_log_cost(
    prob::AbstractDDPProblem,
    params::AbstractDDPParameter,
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
