mutable struct KFProblem
    dt::Float64 # discretization step-size

    nx::Int64 # state dimension
    nu::Int64 # control dimension
    nw::Int64 # process noise dimension
    nv::Int64 # measurement noise dimension

    x_init::AbstractArray{Float64, 1}

    A::AbstractArray{Float64, 2} # State transition matrix
    B::AbstractArray{Float64, 2} # Control matrix
    H::AbstractArray{Float64, 2} # Observation matrix

    Q
    R

    mean
    variance::Float64
    distribution::Normal{Float64}

    function KFProblem(
        dt::Float64,
        nx::Int64,
        nu::Int64,
        nw::Int64,
        nv::Int64,
        x_init::AbstractArray{Float64, 1},
        P::AbstractArray{Float64, 2},
        Q,
        R,
        mean,
        variance::Float64,
        distribution::Normal{Float64},
    )

        new(
            dt,
            nx,
            nu,
            nw,
            nv,
            x_init,
            P,
            Q,
            R,
            mean,
            variance,
            distribution
        )
    end
end

mutable struct EKFProblem
    dt::Float64 # discretization step-size

    dims::ModelDimension

    f # dynamics function
    h # measurement function

    μ_proc
    μ_meas
    Σ_proc
    Σ_meas

    dist_proc
    dist_meas

    function EKFProblem(
        dt::Float64,
        dims::ModelDimension,
        f,
        h,
        μ_proc,
        μ_meas,
        Σ_proc,
        Σ_meas,
    )
        dist_proc = MvNormal(μ_proc, Σ_proc)
        dist_meas = MvNormal(μ_meas, Σ_meas)
        
        new(
            dt,
            dims,
            f,
            h,
            μ_proc,
            μ_meas,
            Σ_proc,
            Σ_meas,
            dist_proc,
            dist_meas,
        )
    end
end


# function KFProblem(
#     model::AbstractDynamicsModel,
# )

#     dt = model.dt
#     tN = model.tN
#     x_dim = model.x_dim
#     u_dim = model.u_dim
#     w_dim = x_dim
#     v_dim = 3

#     x_init = model.x_init
#     # if tr(model.P_cov) == 0
#     #     P = I + zeros(x_dim, x_dim)
#     # else
#     #     P = model.P_cov
#     # end
#     Q_pro = Diagonal(1e+0 * ones(w_dim))
#     R_meas = Diagonal(1e+0 * ones(v_dim))
#     P = I + zeros(x_dim, x_dim)

#     mean_pro = 0.0
#     variance_pro = 1e-7
#     mean_meas = 0.0
#     variance_meas = 1e-4
#     dist_pro = Normal(mean_pro, sqrt(variance_pro))
#     dist_meas = Normal(mean_meas, sqrt(variance_meas))

#     P_arr = zeros(x_dim,x_dim,tN)

#     Q = Q_pro
#     R = R_meas

#     mean = mean_pro
#     variance = variance_pro
#     distribution = dist_pro

#     islinear = false
#     KFProblem(
#         dt,
#         tN,
#         x_dim,
#         u_dim,
#         x_init,
#         P,
#         P_arr,
#         Q,
#         R,
#         mean,
#         variance,
#         distribution,
#         islinear,
#     )
# end

"""
"""
function solve_KF()
end


"""
    solve_EKF()
    
"""
function solve_EKF(
    model::AbstractDynamicsModel, 
    prob::EKFProblem,
    x̂_apr,
    u,
    P̂_apr,
    z;
    dt=prob.dt,
    t=0,
    X_ref=nothing,
    u_md=nothing,
)
    Q = prob.Σ_proc
    R = prob.Σ_meas
    
    p = nothing
    if isnothing(u_md)
        p = ODEParameter(params=model.params.arr, U_ref=u)
    else
        p = ODEParameter(params=model.params.arr, U_ref=u, U_md=u_md)
    end

    x̂ = x̂_apr + rk4_step(prob.f,  x̂_apr, p, t, dt) * dt # state prediction

    if isnothing(u_md)
        ∇ₓf, ∇ᵤf = get_ode_derivatives(prob.f, x̂, u, model.params.arr) # partial derivative of dynamics
    else
        ∇ₓf, ∇ᵤf = get_ode_derivatives(prob.f, x̂, u, model.params.arr, u_md=u_md) # partial derivative of dynamics
    end

    F =  I + ∇ₓf * dt # state transition matrix 
    gw = I + zeros(6,6)
    L = gw
    P̂ = F * P̂_apr * F' + L * Q * L' # covariance matrix prediction
    
    H = get_obs_derivative(prob.h, x̂, zeros(3)) # partial derivative of observation
    y = z - prob.h(x̂, zeros(3)) # measurement residual
    S = H * P̂ * H' + R
    K = P̂ * H' * inv(S) # Kalman gain
    x̂_new = x̂ + K * y # state estimate update
    P̂_new = (I - K * H) * P̂ # covariance estimate update
    return x̂_new, P̂_new
end

function solve_UKF()

end
