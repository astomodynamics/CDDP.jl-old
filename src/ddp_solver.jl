################################################################################
#=
    DDP Problem construction

    This DDP environment is for standard DDP, Constrained DDP (CDDP), 
    Stochastic DDP, iterative LQR, and probabilistic DDP
=#
################################################################################

mutable struct DDPParameter <: AbstractParameter
    reg_param_x::Float64 # regulation parameter for state
    reg_param_u::Float64 # regulation parameter for control
    reg_param_x_fact::Float64 # regulation factor for state
    reg_param_u_fact::Float64 # regulation factor for control
    reg_param_x_lb::Float64 # regulation factor for state lower bound
    reg_param_u_lb::Float64 # regulation factor for control lower bound
    reg_param_x_ub::Float64 # regulation factor for state lower bound
    reg_param_u_ub::Float64 # regulation factor for control lower bound

    line_search_steps::Vector{Float64} 
end

mutable struct CDDPParameter <: AbstractParameter
    reg_param_x::Float64 # regulation parameter for state
    reg_param_u::Float64 # regulation parameter for control
    reg_param_x_fact::Float64 # regulation factor for state
    reg_param_u_fact::Float64 # regulation factor for control
    reg_param_x_lb::Float64 # regulation factor for state lower bound
    reg_param_u_lb::Float64 # regulation factor for control lower bound
    reg_param_x_ub::Float64 # regulation factor for state lower bound
    reg_param_u_ub::Float64 # regulation factor for control lower bound

    line_search_steps::Vector{Float64} 
    μip::Float64 
    μip_lb::Float64
    isfeasible::Bool
end


function ddp_params_update!(ddp_params::AbstractParameter; reg1=true, reg2=true, pert=false, descend=true)
    if descend
        if reg1
            ddp_params.reg_param_x = max(ddp_params.reg_param_x_lb, ddp_params.reg_param_x/ddp_params.reg_param_x_fact)
        end

        if reg2
            ddp_params.reg_param_u = max(ddp_params.reg_param_u_lb, ddp_params.reg_param_u/ddp_params.reg_param_u_fact)
        end

        if pert
            ddp_params.μip = max(ddp_params.μip_lb, min(ddp_params.μip/5, ddp_params.μip^1.2))
        end
    else
        if reg1
            ddp_params.reg_param_x = min(ddp_params.reg_param_x_ub, ddp_params.reg_param_x*ddp_params.reg_param_x_fact)
        end
        
        if reg2
            ddp_params.reg_param_u = min(ddp_params.reg_param_u_ub, ddp_params.reg_param_u*ddp_params.reg_param_u_fact)
        end
    end
end


"""
    solve_ilqr

solve_ilqr is bit faster than solve_ddp with ilqr due to euler integration in forward pass
"""
function solve_ilqr(
    prob::AbstractDDPProblem;
    max_ite::Int64=10,
    tol::Float64=1e-6,
    max_exe_time=200,
    reg_param_x=1e-2,
    reg_param_u=1e-2,
    reg_param_x_fact=10,
    reg_param_u_fact=10,
    reg_param_x_lb=1e-16,
    reg_param_u_lb=1e-16,
    reg_param_x_ub=1e+3,
    reg_param_u_ub=1e+3,
    line_search_steps=5 .^ LinRange(0, -6, 30),
    X=nothing,
    U=nothing,
    randomize=false,
    verbose=true,
)
    @printf("**************************************************************************************\n\
             >>> Start iLQR Problem Solver \n\
            **************************************************************************************\n")

    ddp_params = DDPParameter(
        reg_param_x,
        reg_param_u,
        reg_param_x_fact,
        reg_param_u_fact,
        reg_param_x_lb,
        reg_param_u_lb,
        reg_param_x_ub,
        reg_param_u_ub,
        line_search_steps
    )

    if isequal(X, nothing)
        X, U = initialize_trajectory(prob.model, x_init=prob.x_init, tf=prob.tf, tN=prob.tN, f=prob.f, randomize=false)
    elseif isequal(X, nothing) && !isequal(U, nothing)
        X = simulate_trajectory(prob.model, prob.x_init, U, prob.tf, prob.dt, f=prob.f)
    elseif isequal(U,nothing)
        _, U = initialize_trajectory(prob.model, x_init=prob.x_init, tf=prob.tf, tN=prob.tN, f=prob.f, randomize=false)
    end

    J = get_trajectory_cost(X, U, prob.X_ref, prob.x_final, prob.ell, prob.ϕ, prob.tN, prob.dt) 
    J_old = Inf
    gains = DDPGain([], [])
    sol = DDPSolution(X, U, J, gains)

    success, ite = false, 0
    t_init = time()
    while (ite < max_ite) && !(success && abs(J_old - J)/J < tol)
        if time() - t_init > max_exe_time
            @printf("Maximum computation time passed!!!")
            break
        end

        if verbose
            if (mod(ite, 10) == 0)
                @printf("\
                iter    objective 
                \n")
            end
                @printf("\
                  %d       %.6f,  
                \n", ite, J)
        end

        backward_pass_ilqr!(sol, prob, ddp_params)
        forward_pass_ilqr!(sol, prob, ddp_params)

        if sol.J <= J
            success = true
            sol.X = simulate_trajectory(prob.model, prob.x_init, sol.U, prob.tf, prob.dt, f=prob.f)
            J_old = copy(J)
            J = copy(sol.J)
            ddp_params_update!(ddp_params, reg1=true, reg2=true, descend=true)
        else
            ddp_params_update!(ddp_params, reg1=true, reg2=true, descend=false)
        end
        ite += 1
    end
    # sol.X = simulate_trajectory(prob.model, prob.x_init, sol.U, prob.tf, prob.dt, f=prob.f)

    @printf("**************************************************************************************\n\
             >>> Successfully Finished iLQR Problem Solver <<< \n\
            **************************************************************************************\n")
    return sol
end


function solve_ddp(
    prob::AbstractDDPProblem;
    max_ite::Int64=10,
    tol::Float64=1e-6,
    max_exe_time=120,
    reg_param_x=1e-2,
    reg_param_u=1e-2,
    reg_param_x_fact=10,
    reg_param_u_fact=10,
    reg_param_x_lb=1e-12,
    reg_param_u_lb=1e-12,
    reg_param_x_ub=1e+2,
    reg_param_u_ub=1e+2,
    line_search_steps=5 .^ LinRange(0, -6, 30),
    X=nothing,
    U=nothing,
    randomize=false,
    isilqr=false,
    verbose=true,
)
    @printf("**************************************************************************************\n\
             >>> Start DDP Problem Solver \n\
            **************************************************************************************\n")
        
    ddp_params = DDPParameter(
        reg_param_x,
        reg_param_u,
        reg_param_x_fact,
        reg_param_u_fact,
        reg_param_x_lb,
        reg_param_u_lb,
        reg_param_x_ub,
        reg_param_u_ub,
        line_search_steps
    )

    if isequal(X, nothing)
        X, U = initialize_trajectory(prob.model, x_init=prob.x_init, tf=prob.tf, tN=prob.tN, f=prob.f, randomize=false)
    elseif isequal(X, nothing) && !isequal(U, nothing)
        X = simulate_trajectory(prob.model, prob.x_init, U, prob.tf, prob.dt, f=prob.f)
    elseif isequal(U,nothing)
        _, U = initialize_trajectory(prob.model, x_init=prob.x_init, tf=prob.tf, tN=prob.tN, f=prob.f, randomize=false)
    end
    
    J = get_trajectory_cost(X, U, prob.X_ref, prob.x_final, prob.ell, prob.ϕ, prob.tN, prob.dt) 
    J_old = Inf
    gains = DDPGain([], [])
    sol = DDPSolution(X, U, J, gains)

    success, ite = false, 0
    t_init = time()
    while (ite < max_ite) && !(success && abs(J_old - J)/J < tol)
        if time() - t_init > max_exe_time
            @printf("Maximum computation time passed!!!")
            break
        end

        if verbose
            if (mod(ite, 10) == 0)
                @printf("\
                iter    objective  
                \n")
            end
                @printf("\
                  %d       %.6f,  
                \n", ite, J)
        end
        
        backward_pass_ddp!(sol, prob, ddp_params)
        forward_pass_ddp!(sol, prob, ddp_params)

        if sol.J < J
            success = true
            sol.X = simulate_trajectory(prob.model, prob.x_init, sol.U, prob.tf, prob.dt, f=prob.f)
            J_old = copy(J)
            J = copy(sol.J)
            ddp_params_update!(ddp_params, reg1=true, reg2=true, descend=true)
        else
            ddp_params_update!(ddp_params, reg1=true, reg2=true, descend=false)
        end
        ite += 1
    end

    @printf("**************************************************************************************\n\
             >>> Successfully Finished DDP Problem Solver <<< \n\
            **************************************************************************************\n")
    return sol
end


"""
    solve_cddp(prob, args)
"""
function solve_cddp(
    prob::AbstractDDPProblem;
    max_ite::Int64=10,
    tol::Float64=1e-6,
    max_exe_time=200,
    reg_param1=1e-4,
    reg_param2=1e-2,
    reg_param1_fact=10,
    reg_param2_fact=10,
    reg_param1_lb=1e-12,
    reg_param2_lb=1e-12,
    reg_param1_ub=1e+2,
    reg_param2_ub=1e+2,
    line_search_steps=5 .^ LinRange(0, -5, 30),
    μip=1e-6,
    μip_lb=1e-12,
    isfeasible=false,
    X=nothing,
    U=nothing,
    Λ=nothing,
    randomize=false,
    isilqr=false,
    verbose=true,
)
    if verbose
        @printf("**************************************************************************************\n\
                >>> Start CDDP Problem Solver \n\
                **************************************************************************************\n")
    end
    ddp_params = CDDPParameter(
        reg_param1,
        reg_param2,
        reg_param1_fact,
        reg_param2_fact,
        reg_param1_lb,
        reg_param2_lb,
        reg_param1_ub,
        reg_param2_ub,
        line_search_steps,
        μip,
        μip_lb,
        false,
    )

    if isequal(X, nothing)
        X, U = initialize_trajectory(prob.model, x_init=prob.x_init, tf=prob.tf, tN=prob.tN, f=prob.f, randomize=false)
    elseif isequal(X, nothing) && !isequal(U, nothing)
        X = simulate_trajectory(prob.model, prob.x_init, U, prob.tf, prob.dt, f=prob.f)
    elseif isequal(U,nothing)
        _, U = initialize_trajectory(prob.model, x_init=prob.x_init, tf=prob.tf, tN=prob.tN, f=prob.f, randomize=false)
    end
    
    J = get_trajectory_cost(X, U, nothing, prob.x_final, prob.ell, prob.ϕ, prob.tN, prob.dt) 
    J_old = Inf
    ddp_params.μip = μip 
    isfeasible = get_feasibility(prob, X, U)
    ddp_params.isfeasible = isfeasible
    if isfeasible 
        ddp_params.μip = 1e-8
    end
    println("isfeasible: ", isfeasible)
    Λ_arr::Vector{Vector{Float64}} = Vector[]
    Y_arr::Vector{Vector{Float64}} = Vector[]
    
    for k in 0:prob.tN
        t = k * prob.dt
        if isfeasible
            λ = zeros(prob.λ_dim) 
            y = zeros(prob.λ_dim) 
        else
            λ = 5e+0 * ones(prob.λ_dim)
            y = 1e+0 * ones(prob.λ_dim)
        end
        
        push!(Λ_arr, λ)
        push!(Y_arr, y)
    end
    
    Λ = ConstantInterpolation(Λ_arr, 0:prob.dt:prob.tf)
    Y = ConstantInterpolation(Y_arr, 0:prob.dt:prob.tf)    

    gains = CDDPGain([], [], [], [], [], [])
    sol = CDDPSolution(X, U, Λ, Y, J, gains)

    J = Inf
    J_old = Inf
    
    success, ite = false, 0
    t_init = time()

    while (ite < max_ite) && !(success && abs(J_old - J)/J < tol)
        if time() - t_init > max_exe_time
            @printf("Maximum computation time passed!!!")
            break
        end

        if verbose
            if (mod(ite, 10) == 0)
                @printf("\
                iter    objective   inf_pr     inf_du   lg(mu)   |d|   lg(rg)  alpha_du   alpha_pr   ls
                \n")
            end
                @printf("\
                  %d       %.6f,  
                \n", ite, J)
        end

        backward_pass_cddp!(sol, prob, ddp_params)
        forward_pass_cddp!(sol, prob, ddp_params)

        if sol.J < J
            # find better solution trajectory
            sol.X = simulate_trajectory(prob.model, prob.x_init, sol.U, prob.tf, prob.dt, f=prob.f)
            J_old = copy(J)
            J = copy(sol.J)
            success = true

            ddp_params_update!(ddp_params, reg1=true, reg2=true, pert=true, descend=true)
        else
            ddp_params_update!(ddp_params, reg1=true, reg2=true, pert=false, descend=false)
        end
        
        ite += 1
    end

    if verbose
        @printf("**************************************************************************************\n\
             >>> Successfully Finished CDDP Problem Solver <<< \n\
            **************************************************************************************\n")
    end
    
    return sol
end
