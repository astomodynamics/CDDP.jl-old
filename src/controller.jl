#=
###############################################################################
#    Model Predictive Path Integral Control Algorithm 
###############################################################################
=#

# MPPI solver as a Julia class
mutable struct MPPIProblemCPU <: AbstractMPPIProblem
    model::AbstractDynamicsModel

    # simulation parameters
    tf::Float64 # final time
    tN::Int64 # number of time-discretization steps
    dt::Float64 # discretization step-size

    # dimensions
    dims::ModelDimension # model dimensions

    # cost objective
    cost_funcs::MPPICostFunction # cost functions

    # dynamics
    dyn_funcs::MPPIDynamicsFunction # dynamics model function

    # boundary conditions
    x_init::Vector{Float64} # initial state 
    x_final::Vector{Float64} # terminal state

    X_ref # reference trajectory
    U_md # measured disturbance
    function MPPIProblemCPU(
        model;
        tf=model.tf,
        tN=model.tN,
        dt=model.dt,
        dims=model.dims,
        cost_funcs=empty,
        dyn_funcs=empty,
        x_init=model.x_init,
        x_final=model.x_final,
        X_ref=nothing,
        U_md=nothing,
        ) 
        new(
            model,
            tf,
            tN,
            dt,
            dims,
            cost_funcs,
            dyn_funcs,
            x_init,
            x_final,
            X_ref,
            U_md,
        )
    end
end


"""
    solve_mppi(model, mppi_problem, X0, U0)

Solves the MPPI control problem and returns solution trajectories of state and
control inputs.

# Arguments
- `problem`: MPPI problem definition
- `X`: initial state trajectory
- `U`: initial control trajectory
- `K`: number of samples
- `λ`: temperature in entropy
- `μ`: mean of dynamics noise
- `Σ`: variance of dynamics noise
- `threads`: number of threads for GPU acceleration
- `randomize`: whether to randomize the initial state and control trajectories
- `verbose`: whether to print the progress

# Return
- `X`: state trajectory
- `U`: control trajectory

# Examples
```julia

model = AbstractDynamicsModel()
mppi_problem = MPPIProblem(model, model.dt, model.tN, model.hN)
X, U = solve_mppi_cpu(model, problem)

```
"""

function solve_mppi(
    prob::AbstractMPPIProblem;
    U=nothing,
    K=10,
    λ=100.0,
    μ=0.0,
    Σ=1e-0,
    threads=1,
    randomize=false,
    verbose=false,
    δu=nothing,
)
    if verbose && isequal(typeof(prob), MPPIProblemCPU)
        @printf("Solving MPPI problem on CPU...\n")
        # solve_mppi_cpu(prob, X, U, K, λ, μ, σ, ν, ρ, threads, randomize, verbose)
    elseif verbose && isequal(typeof(prob), MPPIProblemGPU)
        @printf("Solving MPPI problem on GPU...\n")
        # solve_mppi_gpu(prob, X, U, K, λ, μ, σ, ν, ρ, threads, randomize, verbose)
    elseif isequal(prob, nothing)
        error("MPPI problem is not defined.")
    end

    # setup mppi parameters
    cost_funcs = prob.cost_funcs
    dyn_funcs = prob.dyn_funcs
    x_init = prob.x_init
    tf, tN, dt = prob.tf, prob.tN, prob.dt
    nx, nu = prob.dims.nx, prob.dims.nu
    X_ref = prob.X_ref
    params_arr = prob.model.params.arr


    if isequal(U, nothing)
        if randomize
            U = 1e-8 * rand(Float64,(nu, tN)) # use this if you want randomized initial trajectory
        else
            U = zeros(nu, tN)
        end
    end

    S = zeros(K) # entroy cost, trajectory cost

    # setup dynamics noise
    if size(μ, 1) == 1
        dist = Normal(μ, sqrt(Σ))
    else
        dist = MvNormal(μ, Σ)
    end
    
    X_arr = zeros(nx, tN+1) # state trajectory array

    if isnothing(δu)
        δu = rand(dist, 1, tN, K) # sampling noise
    end

    if isnothing(X_ref)
        X_ref = zeros(nx, tN+1) # reference trajectory
    end

    # TODO: implement GPU acceleration or parallelization
    for k in 1:K
        X_arr[:,1] = x_init
        
        for i in 1:tN
            uk = U[:,i] + δu[:, i, k][1]

            # uk = clamp.(uk, -100, 100) # clamp the value for control constraints
            t = (i - 1) * dt
            
            if isnothing(prob.U_md)
                p = ODEParameter(params=params_arr, U_ref=uk)
            else
                p = ODEParameter(params=params_arr, U_ref=uk, U_md=prob.U_md[:,k])
            end
            X_arr[:, i+1] = X_arr[:, i] + rk4_step(dyn_funcs.f, X_arr[:, i], p, t, dt) * dt

            S[k] += cost_funcs.ell(X_arr[:, i], X_ref[:, i]) + λ * U[:, i]' * Σ * δu[:, i, k][1]
        end
        S[k] += cost_funcs.ϕ(X_arr[:, tN+1], X_ref[:, tN+1])
    end

    β = minimum(S)
    η = sum(exp.(-(S .- β) / λ))

    w = zeros(K)
    for k in 1:K
        w[k] = exp(-(S[k] - β) / λ) / η
    end

    # compute control from optimal distribution
    for i in 1:tN
        @printf("δu[:,i,:]: %s\n", δu[:,i,:])
        U[:,i] += δu[:,i,:][1] * w
    end

    u_out = U[:, 1]
    U[:, 1:end-1] = U[:, 2:end]
    U[:, end] = rand(dist, nu)
    return u_out, U
end


function solve_mppi_old(
    prob::AbstractMPPIProblem;
    U=nothing,
    K=10,
    λ=100.0,
    μ=0.0,
    Σ=1e-0,
    threads=1,
    randomize=false,
    verbose=true,
    δu=nothing,
)
    # if verbose && isequal(typeof(prob), MPPIProblemCPU)
    #     @printf("Solving MPPI problem on CPU...\n")
    #     # solve_mppi_cpu(prob, X, U, K, λ, μ, σ, ν, ρ, threads, randomize, verbose)
    # elseif verbose && isequal(typeof(prob), MPPIProblemGPU)
    #     @printf("Solving MPPI problem on GPU...\n")
    #     # solve_mppi_gpu(prob, X, U, K, λ, μ, σ, ν, ρ, threads, randomize, verbose)
    # elseif isequal(prob, nothing)
    #     error("MPPI problem is not defined.")
    # end

    # setup mppi parameters
    cost_funcs = prob.cost_funcs
    dyn_funcs = prob.dyn_funcs
    x_init = prob.x_init
    tf, tN, dt = prob.tf, prob.tN, prob.dt
    nx, nu = prob.dims.nx, prob.dims.nu
    X_ref = prob.X_ref
    params_arr = prob.model.params.arr


    if isequal(U, nothing)
        if randomize
            U = 1e-8 * rand(Float64,(nu, tN)) # use this if you want randomized initial trajectory
        else
            U = zeros(nu, tN)
        end
    end

    S = zeros(K) # entroy cost, trajectory cost

    # setup dynamics noise
    if size(μ, 1) == 1
        dist = Normal(μ, sqrt(Σ))
    else
        dist = MvNormal(μ, Σ)
    end
    
    X_arr = zeros(nx, tN+1) # state trajectory array
    # δu = rand(dist, nu, tN, K) # sampling noise

    # TODO: implement GPU acceleration or parallelization
    for k in 1:K
        X_arr[:,1] = x_init
        
        for i in 1:tN
            uk = U[:,i] + δu[:, i, k]
            
            # uk = clamp.(uk, -100, 100) # clamp the value for control constraints
            t = (i - 1) * dt
            p = ODEParameter(params=params_arr, U_ref=uk)
            X_arr[:, i+1] = X_arr[:, i] + rk4_step(dyn_funcs.f, X_arr[:, i], p, t, dt) * dt

            S[k] += cost_funcs.ell(X_arr[:, i], X_ref[:, i]) + get_sampling_cost_update(prob, U[:,i], δu[:, i, k])
        end
        S[k] += cost_funcs.ϕ(X_arr[:, tN+1], X_ref[:, tN+1])
    end

    # compute control from optimal distribution
    for i in 1:tN
        U[:,i] += get_optimal_distribution(prob, S, δu[:, i, :], λ)
    end

    u_out = U[:, 1]
    U[:, 1:end-1] = U[:, 2:end]
    U[:, end] = rand(dist, nu)
    return u_out, U
end



"""
"""
function simulate_mppi(
    prob::AbstractMPPIProblem;
    U=nothing,
    K=10,
    λ=100.0,
    μ=0.0,
    Σ=1e-0,
    threads=1,
    randomize=false,
    verbose=true,
    δU=nothing,
)
    # if verbose && isequal(typeof(prob), MPPIProblemCPU)
    #     @printf("Solving MPPI problem on CPU...\n")
    #     # solve_mppi_cpu(prob, X, U, K, λ, μ, σ, ν, ρ, threads, randomize, verbose)
    # elseif verbose && isequal(typeof(prob), MPPIProblemGPU)
    #     @printf("Solving MPPI problem on GPU...\n")
    #     # solve_mppi_gpu(prob, X, U, K, λ, μ, σ, ν, ρ, threads, randomize, verbose)
    # elseif isequal(prob, nothing)
    #     error("MPPI problem is not defined.")
    # end

    # setup mppi parameters
    cost_funcs = prob.cost_funcs
    dyn_funcs = prob.dyn_funcs
    x_init = prob.x_init
    tf, tN, dt = prob.tf, prob.tN, prob.dt
    nx, nu = prob.dims.nx, prob.dims.nu
    X_ref = prob.X_ref
    params_arr = prob.model.params_arr

    N = 500

    if isequal(U, nothing)
        if randomize
            U = 1e-8 * rand(Float64,(nu, tN)) # use this if you want randomized initial trajectory
        else
            U = zeros(nu, tN)
        end
    end

    X_sol = zeros(nx, N)
    U_sol = zeros(nu, N)
    X_sol[:,1] = x_init

    # main loop
    for j in 1:N-1
        S = zeros(K) # entroy cost, trajectory cost

        # setup dynamics noise
        # if size(μ, 1) == 1
        #     dist = Normal(μ, sqrt(Σ))
        # else
        #     dist = MvNormal(μ, Σ)
        # end

        # dist = Normal(μ, sqrt(Σ))
        
        X_arr = zeros(nx, tN) # state trajectory array
        # δu = rand(dist, nu, tN, K) # sampling noise
        δu = δU[:,:,:,j]

        # TODO: implement GPU acceleration or parallelization
        for k in 1:K
            X_arr[:,1] = x_init
            
            for i in 1:tN-1
                uk = U[:,i] + δu[:, i, k]
                
                # uk = clamp.(uk, -100, 100) # clamp the value for control constraints
                t = (i - 1) * dt
                p = ODEParameter(params=prob.model.params.arr, U_ref=uk)
                X_arr[:, i+1] = X_arr[:, i] + rk4_step(dyn_funcs.f, X_arr[:, i], p, t, dt) * dt

                S[k] += cost_funcs.ell(X_arr[:, i], X_ref(t))
                # S[k] += cost_funcs.l(X_arr[:, i], X_ref(t)) + get_sampling_cost_update(prob, U[:,i], δu[:, i, k])
                
            end

            # S[k] += cost_funcs.ϕ(X_arr[:, tN+1], X_ref(tf))
        end
        # println(S)

        # compute control from optimal distribution
        for i in 1:tN
            U[:,i] += get_optimal_distribution(prob, S, δu[:, i, :], λ)
        end

        u_out = U[:, 1]
        U[:, 1:end-1] = copy(U[:, 2:end])
        # U[:, end] = rand(dist, nu)

        p = ODEParameter(params=params_arr, U_ref=u_out)
        X_sol[:, j+1] = X_sol[:, j] + rk4_step(dyn_funcs.f, X_sol[:, j], p, 0.0, dt) * dt
        U_sol[:, j] = u_out
        x_init = X_sol[:, j+1]
        
        # return u_out, U, S
    end
    return X_sol, U_sol
end


"""
    get_optimal_distribution(model, problem, S, δu)
 
Compute optimal distribution from relative entropy at a specific time

"""
function get_optimal_distribution(
    problem::AbstractMPPIProblem,
    S::AbstractArray{Float64,1},
    δu::AbstractArray{Float64,2},
    λ::Float64,
)
    # S = S/sum(S) # normalization of cost function if needed
    K = length(S) # number of samples

    sum1 = zeros(size(δu,1))
    sum2 = 0
    for k in 1:K
        sum1 += exp(-(1 / λ) * S[k]) * δu[:, k]
        sum2 += exp(-(1 / λ) * S[k])
    end
    return sum1 ./ sum2
end

"""
    get_sampling_cost_update(model, problem, x, u, δu)

update cost function 
"""
function get_sampling_cost_update( 
    problem::AbstractMPPIProblem,
    u::AbstractArray{Float64,1},
    δu::AbstractArray{Float64,1},
)
    ν = 100.
    R = 1.
    ell = (1 - ν^-1) / 2 * transpose(δu) * R * δu + transpose(u) * R * δu + 1/2 * transpose(u) * R * u
    return ell
end
