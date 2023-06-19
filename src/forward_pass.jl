
"""
    forward_pass_ddp()
"""
function forward_pass_ddp!(
    sol::DDPSolution,
    prob::DDPProblem,
    params::DDPParameter;
    interpolate=ConstantInterpolation,
)   
    X, U, J = sol.X, sol.U, sol.J
    l, L = sol.gains.l, sol.gains.L
    tf, dt, tN = prob.tf, prob.dt, prob.tN
    J = sol.J
    dyn_funcs, cost_funcs = prob.dyn_funcs, prob.cost_funcs

    for step in params.line_search_steps
        x_new = prob.x_init
        X_new::Vector{Vector{Float64}} = Vector[]
        U_new::Vector{Vector{Float64}} = Vector[]
        push!(X_new, x_new)
        
        # TODO: simulate forward pass using DifferentialEquations.jl
        for k in 0:tN-1
            t = k*dt
            # deviation from nominal trajectory
            δx = x_new - X(t)
            
            # update local optimal control
            u_new = U(t) + step * l(t) + L(t) * δx

            # propagate the next state
            p = ODEParameter(params=prob.model.params.arr, U_ref=u_new)

            x_new = dyn_funcs.disc_ode(x_new, p, dt)
            
            # save new trajectory
            push!(X_new, x_new)
            push!(U_new, u_new)
        end
        push!(U_new, zeros(prob.dims.nu)) 

        # convert X and U array to continuous function
        # X_new_func =  interpolate(X_new, 0:dt:tf)
        # U_new_func =  interpolate(U_new, 0:dt:tf)
        X_new_func =  ConstantInterpolation(X_new, 0:dt:tf)
        U_new_func =  ConstantInterpolation(U_new, 0:dt:tf)

        # evaluate the new trajectory
        J_new = get_trajectory_cost(X_new_func, U_new_func, prob.X_ref, prob.x_final, cost_funcs.ell, cost_funcs.ϕ, prob.tN, prob.dt) 

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
    sol::CDDPSolution,
    prob::CDDPProblem,
    params::CDDPParameter;
)   
    X, U, Λ, Y, J = sol.X, sol.U, sol.Λ, sol.Y, sol.J
    l, L = sol.gains.l, sol.gains.L
    m, M, n, N = sol.gains.m, sol.gains.M, sol.gains.n, sol.gains.N
    tf, dt, tN = prob.tf, prob.dt, prob.tN
    
    for step in params.line_search_steps
        x_new = prob.x_init
        X_new::Vector{Vector{Float64}} = Vector[]
        U_new::Vector{Vector{Float64}} = Vector[]
        Λ_new::Vector{Vector{Float64}} = Vector[]
        Y_new::Vector{Vector{Float64}} = Vector[]
        push!(X_new, x_new)

        for k in 0:tN-1
            t = k*dt
            # deviation from nominal trajectory
            δx = x_new - X(t) 
            
            # update local optimal control
            u_new = U(t) + step * l(t) + L(t) * δx

            # update Lagrange multiplier
            λ_new = Λ(t) + step * m(t) + M(t) * δx

            # propagate the next state
            p = ODEParams(prob.model, u_new, isarray=true)
            x_new += rk4_step(prob.f, x_new, p, t, h=dt) * dt

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
        push!(U_new, zeros(prob.u_dim)) 
        push!(Λ_new, zeros(prob.λ_dim))
        push!(Y_new, zeros(prob.λ_dim))
        

        # convert X and U array to continuous function
        X_new_func =  ConstantInterpolation(X_new, 0:dt:tf)
        U_new_func =  ConstantInterpolation(U_new, 0:dt:tf)
        Λ_new_func =  ConstantInterpolation(Λ_new, 0:dt:tf)
        Y_new_func =  ConstantInterpolation(Y_new, 0:dt:tf)

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

