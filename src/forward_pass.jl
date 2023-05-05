"""
    forward_pass_ilqr()
"""
function forward_pass_ilqr!(
    sol::DDPSolutions,
    prob::iLQRProblem,
    params::DDPParameter,
)   
    X, U, J = sol.X, sol.U, sol.J
    l, L = sol.gains.l, sol.gains.L
    dt = prob.dt
    J = sol.J

    for step in params.line_search_steps
        x_new = prob.x_init
        X_new::Vector{Vector{Float64}} = Vector[]
        U_new::Vector{Vector{Float64}} = Vector[]
        push!(X_new, x_new)

        for k in 0:prob.tN-1
            t = k*dt
            # deviation from nominal trajectory
            δx = x_new - X(t) 
            
            # update local optimal control
            u_new = U(t) + step * l(t) + L(t) * δx
            
            # propagate the next state
            p = ODEParams(prob.model, u_new, isarray=true)

            # x_new += euler_step(prob.f!, x_new, p, t, h=dt) * dt
            # x_new += rk2_step(prob.f!, x_new, p, t, h=dt) * dt
            x_new += rk4_step(prob.f!, x_new, p, t, h=dt) * dt
            
            # save new trajectory
            push!(X_new, x_new)
            push!(U_new, u_new)
        end

        # convert X and U array to continuous function
        X_new_func = linear_interpolation((collect(LinRange(0.0, prob.tf, prob.tN+1)),), X_new, extrapolation_bc = Line())
        U_new_func = linear_interpolation((collect(LinRange(0.0, prob.tf, prob.tN)),), U_new, extrapolation_bc = Line())

        # evaluate the new trajectory
        J_new = get_trajectory_cost(X_new_func, U_new_func, nothing, prob.x_final, prob.ell, prob.ϕ, prob.tN, prob.dt) 

        if J_new >= J
            continue
        else
            sol.X = X_new_func
            sol.U = U_new_func
            sol.J = J_new
            return nothing
        end
    end
    nothing
end

"""
    forward_pass_ddp()
"""
function forward_pass_ddp!(
    sol::DDPSolutions,
    prob::AbstractDDPProblem,
    params::DDPParameter,
)   
    X, U, J = sol.X, sol.U, sol.J
    l, L = sol.gains.l, sol.gains.L
    dt = prob.dt
    
    for step in params.line_search_steps
        x_new = prob.x_init
        X_new::Vector{Vector{Float64}} = Vector[]
        U_new::Vector{Vector{Float64}} = Vector[]
        push!(X_new, x_new)
        
        for k in 0:prob.tN-1
            t = k*dt
            # deviation from nominal trajectory
            δx = x_new - X(t) 
            
            # update local optimal control
            u_new = U(t) + step * l(t) + L(t) * δx
            
            # propagate the next state
            p = ODEParams(prob.model, u_new, isarray=true)
            x_new += rk4_step(prob.f!, x_new, p, t, h=dt) * dt

            # save new trajectory
            push!(X_new, x_new)
            push!(U_new, u_new)
        end

        # convert X and U array to continuous function
        X_new_func = linear_interpolation((collect(LinRange(0.0, prob.tf, prob.tN+1)),), X_new, extrapolation_bc = Line())
        U_new_func = linear_interpolation((collect(LinRange(0.0, prob.tf, prob.tN)),), U_new, extrapolation_bc = Line())

        # evaluate the new trajectory
        J_new = get_trajectory_cost(X_new_func, U_new_func, nothing, prob.x_final, prob.ell, prob.ϕ, prob.tN, prob.dt) 

        if J_new >= J
            continue
        else
            sol.X = X_new_func
            sol.U = U_new_func
            sol.J = J_new
            return nothing
        end
    end

    nothing
end

"""
    forward_pass_cddp()
"""
function forward_pass_cddp!(
    sol::DDPSolutions,
    prob::AbstractDDPProblem,
    params::CDDPParameter;
)   
    X, U, Λ, Y, J = sol.X, sol.U, sol.Λ, sol.Y, sol.J
    l, L = sol.gains.l, sol.gains.L
    m, M, n, N = sol.gains.m, sol.gains.M, sol.gains.n, sol.gains.N
    dt = prob.dt
    
    for step in params.line_search_steps
        x_new = prob.x_init
        X_new::Vector{Vector{Float64}} = Vector[]
        U_new::Vector{Vector{Float64}} = Vector[]
        Λ_new::Vector{Vector{Float64}} = Vector[]
        Y_new::Vector{Vector{Float64}} = Vector[]
        push!(X_new, x_new)
        for k in 0:prob.tN-1
            t = k*dt
            # deviation from nominal trajectory
            δx = x_new - X(t) 
            
            # update local optimal control
            u_new = U(t) + step * l(t) + L(t) * δx

            # update Lagrange multiplier
            λ_new = Λ(t) + step * m(t) + M(t) * δx

            # propagate the next state
            p = ODEParams(prob.model, u_new, isarray=true)
            x_new += rk4_step(prob.f!, x_new, p, t, h=dt) * dt

            # save new trajectory
            push!(X_new, x_new)
            push!(U_new, u_new)
            push!(Λ_new, λ_new)

            if !params.isfeasible
                # update perturbation vector
                y_new = Y(t) + step * n(t) + N(t) * δx
                push!(Y_new, y_new)
            else
                push!(Y_new, zeros(prob.λ_dim))
            end
            
        end
        
        # convert X and U array to continuous function
        X_new_func = linear_interpolation((collect(LinRange(0.0, prob.tf, prob.tN+1)),), X_new, extrapolation_bc = Line())
        U_new_func = linear_interpolation((collect(LinRange(0.0, prob.tf, prob.tN)),), U_new, extrapolation_bc = Line())
        Λ_new_func = linear_interpolation((collect(LinRange(0.0, prob.tf, prob.tN)),), Λ_new, extrapolation_bc = Line())
        Y_new_func = linear_interpolation((collect(LinRange(0.0, prob.tf, prob.tN)),), Y_new, extrapolation_bc = Line())

        # evaluate the new trajectory
        
        isfeasible_new = get_feasibility(prob, X_new_func, U_new_func)
        
        if isfeasible_new
            J_new = get_trajectory_log_cost(prob, params, X_new_func, U_new_func, Y_new_func, params.isfeasible)
        else
            J_new = J
        end

        # J_new = get_trajectory_cost(X_new_func, U_new_func, nothing, prob.x_final, prob.ell, prob.ϕ, prob.tN, prob.dt) 
        if isfeasible_new && J_new < J
            sol.X = X_new_func
            sol.U = U_new_func
            sol.Λ = Λ_new_func
            sol.Y = Y_new_func
            sol.J = J_new
            return nothing
        end
    end

    nothing
end

