

"""
    backward_pass_ddp()
"""
function backward_pass_ddp!(
    sol::DDPSolution,
    prob::DDPProblem,
    params::DDPParameter;
    isilqr::Bool=false,
    interpolate=ConstantInterpolation
)
    dyn_funcs = prob.dyn_funcs
    cost_funcs = prob.cost_funcs
    X, U = sol.X, sol.U
    tf, dt, tN = prob.tf, prob.dt, prob.tN
    reg_param_x = params.reg_param_x
    reg_param_u = params.reg_param_u

    # initialize arrays for backward pass
    ∇ₓf_arr::Vector{Matrix{Float64}} = Vector[]
    ∇ᵤf_arr::Vector{Matrix{Float64}} = Vector[]
    ∇ₓₓf_arr::Vector{AbstractArray{Float64,3}} = Vector[]
    ∇ₓᵤf_arr::Vector{AbstractArray{Float64,3}} = Vector[]
    ∇ᵤᵤf_arr::Vector{AbstractArray{Float64,3}} = Vector[]

    ell_arr::Vector{Float64} = Vector[]
    ∇ₓell_arr::Vector{Vector{Float64}} = Vector[]
    ∇ᵤell_arr::Vector{Vector{Float64}} = Vector[]
    ∇ₓₓell_arr::Vector{Matrix{Float64}} = Vector[]
    ∇ₓᵤell_arr::Vector{Matrix{Float64}} = Vector[]
    ∇ᵤᵤell_arr::Vector{Matrix{Float64}} = Vector[]

    l_arr::Vector{Vector{Float64}} = Vector[]
    L_arr::Vector{Matrix{Float64}} = Vector[]

    # store dynamics and cost information
    for k in 0:tN-1
        t = k * dt
        x, u = X(t), U(t)
        x_ref = nothing
        u_md = nothing
        p = nothing

        # get reference trajectory
        if !isnothing(prob.X_ref)
            x_ref = prob.X_ref(t)
        else
            x_ref=zeros(length(x))
        end

        # get model disturbance
        if !isnothing(prob.U_md)
            u_md = prob.U_md(t)
        else
            u_md=zeros(length(x))
        end

        # get dynamics information
        if isequal(dyn_funcs.∇f, empty)
            if isilqr
                ∇ₓf, ∇ᵤf = get_ode_derivatives(prob, x, u, isilqr=isilqr)
    
                # store dynamics information
                push!(∇ₓf_arr, I + ∇ₓf * dt)
                push!(∇ᵤf_arr, ∇ᵤf * dt)
            else
                ∇ₓf, ∇ᵤf, ∇ₓₓf, ∇ₓᵤf, ∇ᵤᵤf = get_ode_derivatives(prob, x, u, isilqr=isilqr)
    
                # store dynamics information
                push!(∇ₓf_arr, I + ∇ₓf * dt)
                push!(∇ᵤf_arr, ∇ᵤf * dt)
                push!(∇ₓₓf_arr, ∇ₓₓf * dt)
                push!(∇ₓᵤf_arr, ∇ₓᵤf * dt)
                push!(∇ᵤᵤf_arr, ∇ᵤᵤf * dt)
            end
        else
            p = ODEParameter(params=prob.model.params.arr, U_ref=u, U_md=u_md)
            
            ∇ₓf, ∇ᵤf = dyn_funcs.∇f(x, p)

            # store dynamics information
            push!(∇ₓf_arr, I + ∇ₓf * dt)
            push!(∇ᵤf_arr, ∇ᵤf * dt)

            if !isilqr
                ∇ₓf, ∇ᵤf, ∇ₓₓf, ∇ₓᵤf, ∇ᵤᵤf = dyn_funcs.∇²f(x, p)
    
                # store dynamics information
                push!(∇ₓf_arr, I + ∇ₓf * dt)
                push!(∇ᵤf_arr, ∇ᵤf * dt)
                push!(∇ₓₓf_arr, ∇ₓₓf * dt)
                push!(∇ₓᵤf_arr, ∇ₓᵤf * dt)
                push!(∇ᵤᵤf_arr, ∇ᵤᵤf * dt)
            end
        end
        
        # get cost information
        ell = cost_funcs.ell(x, u, x_ref=x_ref)
        
        if isequal(cost_funcs.∇ell, empty)
            ∇ₓell, ∇ᵤell, ∇ₓₓell, ∇ₓᵤell, ∇ᵤᵤell = get_instant_cost_derivatives(cost_funcs.ell, x, u, x_ref)
        else
            ∇ₓell, ∇ᵤell = cost_funcs.∇ell(x, u, x_ref=x_ref)
            ∇ₓₓell, ∇ₓᵤell, ∇ᵤᵤell = cost_funcs.∇²ell(x, u, x_ref=x_ref)
        end

        # store cost information
        push!(ell_arr, ell*dt)
        push!(∇ₓell_arr, ∇ₓell*dt)
        push!(∇ᵤell_arr, ∇ᵤell*dt)
        push!(∇ₓₓell_arr, ∇ₓₓell*dt)
        push!(∇ₓᵤell_arr, ∇ₓᵤell*dt)
        push!(∇ᵤᵤell_arr, ∇ᵤᵤell*dt)
    end

    ϕ = cost_funcs.ϕ(X(prob.tf), x_ref=prob.x_final)

    if isequal(cost_funcs.∇ϕ, empty)
        ∇ₓϕ, ∇ₓₓϕ = get_terminal_cost_derivatives(cost_funcs.ϕ, X(prob.tf), prob.x_final)
    else
        ∇ₓϕ = cost_funcs.∇ϕ(X(prob.tf), x_ref=prob.x_final)
        ∇ₓₓϕ = cost_funcs.∇²ϕ(X(prob.tf), x_ref=prob.x_final)
    end

    # value function and its derivatives
    V = copy(ϕ)
    ∇ₓV = copy(∇ₓϕ)
    ∇ₓₓV = copy(∇ₓₓϕ)

    push!(l_arr, zeros(prob.dims.nu))
    push!(L_arr, zeros(prob.dims.nu, prob.dims.nx))

    # backward pass
    for k in length(ell_arr):-1:1
        println(∇ₓf_arr[k])
        println(∇ᵤf_arr[k])
        println(∇ₓV)
        # Q = ell_arr[k] + V
        ∇ₓQ = ∇ₓell_arr[k] + ∇ₓf_arr[k]' * ∇ₓV
        ∇ᵤQ = ∇ᵤell_arr[k] + ∇ᵤf_arr[k]' * ∇ₓV

        ∇ₓₓQ = ∇ₓₓell_arr[k] + ∇ₓf_arr[k]' * (∇ₓₓV + reg_param_x * I) * ∇ₓf_arr[k]
        ∇ₓᵤQ = ∇ₓᵤell_arr[k] + ∇ₓf_arr[k]' * (∇ₓₓV + reg_param_x * I) * ∇ᵤf_arr[k]
        ∇ᵤᵤQ = ∇ᵤᵤell_arr[k] + ∇ᵤf_arr[k]' * (∇ₓₓV + reg_param_x * I) * ∇ᵤf_arr[k] + reg_param_u * I

        if !isilqr
            for j = 1:prob.x_dim
                ∇ₓₓQ += ∇ₓV[j] .* ∇ₓₓf_arr[k][:, :, j]
                ∇ₓᵤQ += ∇ₓV[j] .* ∇ₓᵤf_arr[k][:, :, j]
                ∇ᵤᵤQ += ∇ₓV[j] .* ∇ᵤᵤf_arr[k][:, :, j]
            end
        end

        gains_mat = -∇ᵤᵤQ \ [∇ᵤQ  ∇ₓᵤQ'] 
        # compute feedback and feedforward gains
        l = gains_mat[:, 1]
        L = gains_mat[:, 2:end]

        # update values of gradient and hessian of the value function
        # V += 0.5 * l' * ∇ᵤᵤQ * l + l' * ∇ᵤQ
        ∇ₓV = ∇ₓQ + L' * ∇ᵤQ + L' * ∇ᵤᵤQ * l + ∇ₓᵤQ * l
        ∇ₓₓV = ∇ₓₓQ + L' * ∇ₓᵤQ' + ∇ₓᵤQ * L + L' * ∇ᵤᵤQ * L

        # store feedforward and feedback gains
        push!(l_arr, l)
        push!(L_arr, L)
    end
    l_func = interpolate(reverse(l_arr), 0:dt:tf)
    L_func = interpolate(reverse(L_arr), 0:dt:tf)
    sol.gains.l = l_func
    sol.gains.L = L_func
end


"""
    backward_pass_cddp()
"""
function backward_pass_cddp!(
    sol::CDDPSolution,
    prob::CDDPProblem,
    params::CDDPParameter;
    isilqr::Bool=false
)
    X, U, Λ, Y = sol.X, sol.U, sol.Λ, sol.Y
    tf, dt, tN = prob.tf, prob.dt, prob.tN
    reg_param_x = params.reg_param_x
    reg_param_u = params.reg_param_u
    μip = params.μip

    # initialize arrays for backward pass
    ∇ₓf_arr::Vector{Matrix{Float64}} = Vector[]
    ∇ᵤf_arr::Vector{Matrix{Float64}} = Vector[]
    ∇ₓₓf_arr::Vector{AbstractArray{Float64,3}} = Vector[]
    ∇ₓᵤf_arr::Vector{AbstractArray{Float64,3}} = Vector[]
    ∇ᵤᵤf_arr::Vector{AbstractArray{Float64,3}} = Vector[]

    ell_arr::Vector{Float64} = Vector[]
    ∇ₓell_arr::Vector{Vector{Float64}} = Vector[]
    ∇ᵤell_arr::Vector{Vector{Float64}} = Vector[]
    ∇ₓₓell_arr::Vector{Matrix{Float64}} = Vector[]
    ∇ₓᵤell_arr::Vector{Matrix{Float64}} = Vector[]
    ∇ᵤᵤell_arr::Vector{Matrix{Float64}} = Vector[]

    l_arr::Vector{Vector{Float64}} = Vector[]
    L_arr::Vector{Matrix{Float64}} = Vector[]
    m_arr::Vector{Vector{Float64}} = Vector[]
    M_arr::Vector{Matrix{Float64}} = Vector[]
    n_arr::Vector{Vector{Float64}} = Vector[]
    N_arr::Vector{Matrix{Float64}} = Vector[]

    # store dynamics and cost information
    for k in 0:prob.tN-1
        t = k * dt
        x, u = X(t), U(t)

        # get dynamics information
        ∇ₓf, ∇ᵤf, ∇ₓₓf, ∇ₓᵤf, ∇ᵤᵤf = nothing, nothing, nothing, nothing, nothing
        if isequal(prob.∇f, empty)
            if isilqr
                ∇ₓf, ∇ᵤf = get_ode_derivatives(prob, x, u, t, isilqr=isilqr)
    
                # store dynamics information
                push!(∇ₓf_arr, I + ∇ₓf * dt)
                push!(∇ᵤf_arr, ∇ᵤf * dt)
            else
                ∇ₓf, ∇ᵤf, ∇ₓₓf, ∇ₓᵤf, ∇ᵤᵤf = get_ode_derivatives(prob, x, u, t, isilqr=isilqr)
    
                # store dynamics information
                push!(∇ₓf_arr, I + ∇ₓf * dt)
                push!(∇ᵤf_arr, ∇ᵤf * dt)
                push!(∇ₓₓf_arr, ∇ₓₓf * dt)
                push!(∇ₓᵤf_arr, ∇ₓᵤf * dt)
                push!(∇ᵤᵤf_arr, ∇ᵤᵤf * dt)
            end
        else
            if isilqr
                ∇ₓf, ∇ᵤf = prob.∇f(x, u, t)
    
                # store dynamics information
                push!(∇ₓf_arr, I + ∇ₓf * dt)
                push!(∇ᵤf_arr, ∇ᵤf * dt)
            else
                ∇ₓf, ∇ᵤf, ∇ₓₓf, ∇ₓᵤf, ∇ᵤᵤf = prob.∇f(x, u, t)
    
                # store dynamics information
                push!(∇ₓf_arr, I + ∇ₓf * dt)
                push!(∇ᵤf_arr, ∇ᵤf * dt)
                push!(∇ₓₓf_arr, ∇ₓₓf * dt)
                push!(∇ₓᵤf_arr, ∇ₓᵤf * dt)
                push!(∇ᵤᵤf_arr, ∇ᵤᵤf * dt)
            end
        end

        # get cost information
        ell = prob.ell(x, u, prob.x_final)
        ∇ₓell, ∇ᵤell, ∇ₓₓell, ∇ₓᵤell, ∇ᵤᵤell = get_instant_cost_derivatives(prob.ell, x, u, prob.x_final)

        # store cost information
        push!(ell_arr, ell*dt)
        push!(∇ₓell_arr, ∇ₓell*dt)
        push!(∇ᵤell_arr, ∇ᵤell*dt)
        push!(∇ₓₓell_arr, ∇ₓₓell*dt)
        push!(∇ₓᵤell_arr, ∇ₓᵤell*dt)
        push!(∇ᵤᵤell_arr, ∇ᵤᵤell*dt)
    end

    ϕ = prob.ϕ(X(prob.tf), prob.x_final)
    ∇ₓϕ, ∇ₓₓϕ = get_terminal_cost_derivatives(prob.ϕ, X(prob.tf), prob.x_final)

    # value function and its derivatives
    V = copy(ϕ)
    ∇ₓV = copy(∇ₓϕ)
    ∇ₓₓV = copy(∇ₓₓϕ)
    
    push!(l_arr, zeros(prob.u_dim))
    push!(L_arr, zeros(prob.u_dim, prob.x_dim))
    push!(m_arr, zeros(prob.λ_dim))
    push!(M_arr, zeros(prob.λ_dim, prob.x_dim))
    push!(n_arr, zeros(prob.λ_dim))
    push!(N_arr, zeros(prob.λ_dim, prob.x_dim))    

    # backward pass
    for k in length(ell_arr):-1:1
        t = k * dt
        x, u, λ, y = X(t), U(t), Λ(t), Y(t)

        # Q = ell_arr[k] + V
        ∇ₓQ = ∇ₓell_arr[k] + ∇ₓf_arr[k]' * ∇ₓV
        ∇ᵤQ = ∇ᵤell_arr[k] + ∇ᵤf_arr[k]' * ∇ₓV

        ∇ₓₓQ = ∇ₓₓell_arr[k] + ∇ₓf_arr[k]' * (∇ₓₓV + reg_param_x * I) * ∇ₓf_arr[k]
        ∇ₓᵤQ = ∇ₓᵤell_arr[k] + ∇ₓf_arr[k]' * (∇ₓₓV + reg_param_x * I) * ∇ᵤf_arr[k]
        ∇ᵤᵤQ = ∇ᵤᵤell_arr[k] + ∇ᵤf_arr[k]' * (∇ₓₓV + reg_param_x * I) * ∇ᵤf_arr[k] + reg_param_u * I

        if !isilqr
            for j = 1:prob.x_dim
                ∇ₓₓQ += ∇ₓV[j] .* ∇ₓₓf_arr[k][j, :, :]
                ∇ₓᵤQ += ∇ₓV[j] .* ∇ₓᵤf_arr[k][j, :, :]
                ∇ᵤᵤQ += ∇ₓV[j] .* ∇ᵤᵤf_arr[k][j, :, :]
            end
        end
        
        c = prob.c(x, u)
        ∇ₓc, ∇ᵤc, ∇ₓₓc, ∇ₓᵤc, ∇ᵤᵤc = get_instant_const_derivative(prob.c, x, u)
        ∇λₓQ = copy(∇ₓc)
        ∇λᵤQ = copy(∇ᵤc)

        gains_mat = Matrix[]

        if params.isfeasible
            Diag_c = diagm(c) # diagonalize constraint functions
            Diag_λ = diagm(λ) # diagonalize Lagrange multiplier
            r = Diag_λ * c .+ μip # compute the remaining value
            Diag_c_inv = inv(Diag_c) # compute inverse of constraint matrix
            Diag_cInv_Diag_λ = Diag_c_inv * Diag_λ # compute multiplication of inverse and diagonal matrices
            ∇ᵤᵤQ -= ∇λᵤQ' * Diag_cInv_Diag_λ * ∇λᵤQ

            # action-value function update for constrained problem
            ∇ₓQ -= ∇λₓQ' * Diag_c_inv * r
            ∇ᵤQ -= ∇λᵤQ' * Diag_c_inv * r
            ∇ₓₓQ -= ∇λₓQ' * Diag_cInv_Diag_λ * ∇λₓQ
            ∇ₓᵤQ -= ∇λₓQ' * Diag_cInv_Diag_λ * ∇λᵤQ
            ∇ᵤᵤQ -= ∇λᵤQ' * Diag_cInv_Diag_λ * ∇λᵤQ

            gains_mat = -∇ᵤᵤQ \ [∇ᵤQ - ∇λᵤQ' * Diag_c_inv * r  ∇ₓᵤQ' - ∇λᵤQ' * Diag_cInv_Diag_λ * ∇λₓQ] 

            # compute feedback and feedforward gains
            l = gains_mat[:, 1]
            L = gains_mat[:, 2:end]
            m = Diag_c_inv * (r + Diag_λ * ∇λᵤQ * l)
            M = Diag_cInv_Diag_λ * (∇λₓQ + ∇λᵤQ * L)

            # update values of gradient and hessian of the value function
            # V += 0.5 * l' * ∇ᵤᵤQ * l + l' * ∇ᵤQ
            ∇ₓV = ∇ₓQ + L' * ∇ᵤQ + L' * ∇ᵤᵤQ * l + ∇ₓᵤQ * l
            ∇ₓₓV = ∇ₓₓQ + L' * ∇ₓᵤQ' + ∇ₓᵤQ * L + L' * ∇ᵤᵤQ * L

            # store feedforward and feedback gains
            push!(l_arr, l)
            push!(L_arr, L)
            push!(m_arr, m)
            push!(M_arr, M)
            push!(n_arr, zeros(prob.λ_dim))
            push!(N_arr, zeros(prob.λ_dim, prob.x_dim))
        else
            Diag_y = diagm(y) # diagonalize constraint functions
            Diag_λ = diagm(λ) # diagonalize Lagrange multiplier
            r = Diag_λ * y .- μip # compute the remaining value
            r̂ = Diag_λ * (c + y) .- r
            Diag_y_inv = inv(Diag_y) # compute inverse of constraint matrix
            Diag_yInv_Diag_λ = Diag_y_inv * Diag_λ # compute multiplication of inverse and diagonal matrices
            ∇ᵤᵤQ += ∇λᵤQ' * Diag_yInv_Diag_λ * ∇λᵤQ

            # action-value function update for constrained problem
            ∇ₓQ += ∇λₓQ' * Diag_y_inv * r̂
            ∇ᵤQ += ∇λᵤQ' * Diag_y_inv * r̂
            ∇ₓₓQ += ∇λₓQ' * Diag_yInv_Diag_λ * ∇λₓQ
            ∇ₓᵤQ += ∇λₓQ' * Diag_yInv_Diag_λ * ∇λᵤQ
            ∇ᵤᵤQ += ∇λᵤQ' * Diag_yInv_Diag_λ * ∇λᵤQ

            gains_mat = -∇ᵤᵤQ \ [∇ᵤQ + ∇λᵤQ' * Diag_y_inv * r̂  ∇ₓᵤQ' + ∇λᵤQ' * Diag_yInv_Diag_λ * ∇λₓQ] 

            # compute feedback and feedforward gains
            l = gains_mat[:, 1]
            L = gains_mat[:, 2:end]
            m = Diag_y_inv * (r̂ + Diag_λ * ∇λᵤQ * l)
            M = Diag_yInv_Diag_λ * (∇λₓQ + ∇λᵤQ * L)
            n = -(c + y) - ∇λᵤQ * l
            N = -∇λₓQ - ∇λᵤQ * L
            
            # update values of gradient and hessian of the value function
            # V += 0.5 * l' * ∇ᵤᵤQ * l + l' * ∇ᵤQ
            ∇ₓV = ∇ₓQ + L' * ∇ᵤQ + L' * ∇ᵤᵤQ * l + ∇ₓᵤQ * l
            ∇ₓₓV = ∇ₓₓQ + L' * ∇ₓᵤQ' + ∇ₓᵤQ * L + L' * ∇ᵤᵤQ * L

            # store feedforward and feedback gains
            push!(l_arr, l)
            push!(L_arr, L)
            push!(m_arr, m)
            push!(M_arr, M)
            push!(n_arr, n)
            push!(N_arr, N)
        end
    end

    l_func = ConstantInterpolation(reverse(l_arr), 0:dt:tf)
    L_func = ConstantInterpolation(reverse(L_arr), 0:dt:tf)
    m_func = ConstantInterpolation(reverse(m_arr), 0:dt:tf)
    M_func = ConstantInterpolation(reverse(M_arr), 0:dt:tf)
    n_func = ConstantInterpolation(reverse(n_arr), 0:dt:tf)
    N_func = ConstantInterpolation(reverse(N_arr), 0:dt:tf)
    sol.gains.l = l_func
    sol.gains.L = L_func
    sol.gains.m = m_func
    sol.gains.M = M_func
    sol.gains.n = n_func
    sol.gains.N = N_func
    nothing
end


function backward_pass_ilqg!()
    nothing
end

