mutable struct KFProblem
    dt::Float64 # discretization step-size

    nx::Int64 # state dimension
    nu::Int64 # control dimension

    x_init::AbstractArray{Float64, 1}

    A::AbstractArray{Float64, 2} # State transition matrix
    B::AbstractArray{Float64, 2} # Control matrix
    H::AbstractArray{Float64, 2} # Observation matrix
    P::AbstractArray{Float64, 2} # Covariance matrix

    Q
    R

    mean
    variance::Float64
    distribution::Normal{Float64}

    function KFProblem(
        dt::Float64,
        nx::Int64,
        nu::Int64,
        x_init::AbstractArray{Float64, 1},
        P::AbstractArray{Float64, 2},
        Q,
        R,
        mean,
        variance::Float64,
        distribution::Normal{Float64},
        islinear::Bool=false,
    )

        new(
            dt,
            nx,
            nu,
            x_init,
            P,
            Q,
            R,
            mean,
            variance,
            distribution,
            islinear,
        )
    end
end

mutable struct EKFProblem
    function EKFProblem()
        new()
    end
end


mutable struct KFProblem
    dt::Float64 # discretization step-size
    tN::Int64 # number of time-discretization steps

    x_dim::Int64 # state dimension
    u_dim::Int64 # control dimension

    x_init::AbstractArray{Float64, 1}
    
    P::AbstractArray{Float64, 2} # Covariance matrix
    P_arr::AbstractArray{Float64, 3} # Covariance matrix history

    Q
    R

    mean
    variance::Float64
    distribution::Normal{Float64}

    islinear::Bool # used to improve convergence speed by discarding numerical fidelity
end

function KFProblem(
    model::AbstractDynamicsModel,
)

    dt = model.dt
    tN = model.tN
    x_dim = model.x_dim
    u_dim = model.u_dim
    w_dim = x_dim
    v_dim = 3

    x_init = model.x_init
    # if tr(model.P_cov) == 0
    #     P = I + zeros(x_dim, x_dim)
    # else
    #     P = model.P_cov
    # end
    Q_pro = Diagonal(1e+0 * ones(w_dim))
    R_meas = Diagonal(1e+0 * ones(v_dim))
    P = I + zeros(x_dim, x_dim)

    mean_pro = 0.0
    variance_pro = 1e-7
    mean_meas = 0.0
    variance_meas = 1e-4
    dist_pro = Normal(mean_pro, sqrt(variance_pro))
    dist_meas = Normal(mean_meas, sqrt(variance_meas))

    P_arr = zeros(x_dim,x_dim,tN)

    Q = Q_pro
    R = R_meas

    mean = mean_pro
    variance = variance_pro
    distribution = dist_pro

    islinear = false
    KFProblem(
        dt,
        tN,
        x_dim,
        u_dim,
        x_init,
        P,
        P_arr,
        Q,
        R,
        mean,
        variance,
        distribution,
        islinear,
    )
end

"""
"""
function solve_KF()
end


"""
    solve_EKF()
    
"""
function solve_EKF(
    model::AbstractDynamicsModel, 
    problem::KFProblem,
    x̂_apr,
    u,
    P̂_apr,
    z;
    dt=problem.dt,
    t=0,
    h! =model.h!,
)
    Q = problem.Q
    R = problem.R

    p = ODEParams(prob.model, u, isarray=true)
    x̂ = x̂_apr + rk4_step(prob.f!,  x̂_apr, p, t, h=dt) * dt # state prediction
    ∇ₓf, ∇ᵤf = get_ode_derivatives(prob, x, u, t, isilqr=true)
    F =  I + ∇ₓf * dt # state transition matrix 
    gw = I + zeros(6,6)
    L = gw
    P̂ = F * P̂_apr * F' + L * Q * L' # covariance matrix prediction
    
    H = get_obs_derivatives(h!,x̂,t) # partial derivative of observation
    y = z - h!(zeros(3), x̂, t) # measurement residual
    S = H * P̂ * H' + R
    K = P̂ * H' * inv(S) # Kalman gain
    x̂_new = x̂ + K * y # state estimate update
    P̂_new = (I - K * H) * P̂ # covariance estimate update
    return x̂_new, P̂_new
end

function solve_UKF()
end
