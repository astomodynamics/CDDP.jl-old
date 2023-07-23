

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
    model::AbstractDynamicsModel,
    tf::Float64,
    tN::Int64,
    x_init::Vector{Float64},
    f::Function;
    F::Function=empty,
    U_md=nothing,
    X_ref=nothing,
    ode_alg=RK4(),
    sde_alg=EM(),
    reltol=1e-8, 
    abstol=1e-8,
    randomize::Bool=false,
    isstochastic::Bool=false,
)
    dt = tf/tN
    # initialize U array
    if randomize
        U_mat = 1e-9 * rand(Float64,(tN+1, model.dims.nu)) # use this if you want randomized initial trajectory
    else
        U_mat = zeros(tN+1, model.dims.nu)
    end

    U_vec = Vector[U_mat[t, :] for t in axes(U_mat,1)]
    # convert U array into U as a continuous function
    U = ConstantInterpolation(U_vec, 0:dt:tf)
    
    if !isstochastic
        # integrate through DifferentialEquations.jl
        # set ODE parameters (for control and storaged trajectory)
        if isnothing(X_ref)
            if isnothing(U_md)
                p = ODEParameter(params=model.params.arr, U_ref=U)
            else
                p = ODEParameter(params=model.params.arr, U_ref=U, U_md=U_md)
            end
        else
            p = ODEParameter(params=model.params.arr, U_ref=U, X_ref=X_ref)
        end

        # define ODE problem
        prob = ODEProblem(f, x_init, (0.0,tf), p, dt=dt)

        # solve ODE problem
        X = solve(prob, ode_alg, reltol=reltol, abstol=abstol)
    else
        # integrate through DifferentialEquations.jl
        # set ODE parameters (for control and storaged trajectory)
        if isnothing(X_ref)
            if isnothing(U_md)
                p = ODEParameter(params=model.params.arr, U_ref=U)
            else
                p = ODEParameter(params=model.params.arr, U_ref=U, U_md=U_md)
            end
        else
            p = ODEParameter(params=model.params.arr, U_ref=U, X_ref=X_ref)
        end

        # define SDE problem
        prob = SDEProblem(f, F, x_init, (0.0, tf), p, noise=WienerProcess(0.0, 0.0, 0.0))
        
        # solve SDE problem
        X = solve(prob, sde_alg, dt=dt)
    end
    return X, U
end



"""


simulate dynamics given initial condition and control sequence

# Arguments

"""
function simulate_trajectory(
    model::AbstractDynamicsModel,
    tf::Float64,
    dt::Float64,
    x_init::Vector{Float64}, 
    f::Function,
    U;
    F::Function=empty,
    U_md=nothing,
    X_ref=nothing,
    l=nothing,
    L=nothing,
    isfeedback::Bool=false,
    ode_alg=Tsit5(),
    sde_alg=EM(),
    reltol=1e-12, 
    abstol=1e-12,
    isstochastic::Bool=false,
)   

    if !isstochastic
        # integrate through DifferentialEquations.jl
        # set ODE parameters (for control and storaged trajectory)

        if isnothing(X_ref)
            if isnothing(U_md)
                p = ODEParameter(params=model.params.arr, U_ref=U)
            else
                p = ODEParameter(params=model.params.arr, U_ref=U, U_md=U_md)
            end
        else
            if !isfeedback
                p = ODEParameter(params=model.params.arr, U_ref=U, X_ref=X_ref)
            else
                p = ODEParameter(params=model.params.arr, U_ref=U, X_ref=X_ref, l=l, L=L)
            end
        end

        # define ODE problem
        prob = ODEProblem(f, x_init, (0.0,tf), p)


        # solve ODE problem
        X = solve(prob, ode_alg, reltol=reltol, abstol=abstol)
    else
        # integrate through DifferentialEquations.jl
        # set ODE parameters (for control and storaged trajectory)
        if isnothing(X_ref)
            if isnothing(U_md)
                p = ODEParameter(params=model.params.arr, U_ref=U)
            else
                p = ODEParameter(params=model.params.arr, U_ref=U, U_md=U_md)
            end
        else
            if !isfeedback
                p = ODEParameter(params=model.params.arr, U_ref=U, X_ref=X_ref)
            else
                p = ODEParameter(params=model.params.arr, U_ref=U, X_ref=X_ref, l=l, L=L)
            end
        end

        # define SDE problem
        prob = SDEProblem(f, F, x_init, (0.0, tf), p, noise=WienerProcess(0.0, 0.0, 0.0))
        
        # solve SDE problem
        X = solve(prob, sde_alg, dt=dt)
    end

    return X
end