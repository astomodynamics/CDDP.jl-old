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
    ∇f # dynamics function derivative
    h # measurement function
    ∇h # measurement function derivative

    μ_proc
    μ_meas
    Σ_proc
    Σ_meas

    function EKFProblem(
        dt::Float64,
        dims::ModelDimension,
        f,
        ∇f,
        h,
        ∇h,
        μ_proc,
        μ_meas,
        Σ_proc,
        Σ_meas,
    )

        new(
            dt,
            dims,
            f,
            ∇f,
            h,
            ∇h,
            μ_proc,
            μ_meas,
            Σ_proc,
            Σ_meas,
        )
    end
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
    nx, nu, nw, nv = model.dims.nx, model.dims.nu, model.dims.nw, model.dims.nv
    Q = prob.Σ_proc
    R = prob.Σ_meas
    
    p = nothing
    if isnothing(u_md)
        p = ODEParameter(params=model.params.arr, U_ref=u)
    else
        p = ODEParameter(params=model.params.arr, U_ref=u, U_md=u_md)
    end

    x̂ = x̂_apr + rk4_step(prob.f,  x̂_apr, p, t, dt) * dt # state prediction

    ∇ₓf = zeros(nx,nx)
    if isnothing(u_md)
        if isnothing(prob.∇f) 
            ∇ₓf, _ = get_ode_derivatives(prob.f, x̂_apr, u, model.params.arr) # partial derivative of dynamics
        else
            ∇ₓf, _ = prob.∇f(x̂_apr, p)
        end
        ∇ₓf, _ = get_ode_derivatives(prob.f, x̂_apr, u, model.params.arr) # partial derivative of dynamics
        # @printf("∇ₓf : %s\n", ∇ₓf)
    else
        if isnothing(prob.∇f) 
            ∇ₓf, _ = get_ode_derivatives(prob.f, x̂_apr, u, model.params.arr, u_md=u_md) # partial derivative of dynamics
        else
            ∇ₓf, _ = prob.∇f(x̂_apr, p)
        end
        # @printf("∇ₓf : %s\n", ∇ₓf)
    end



    F =  I + ∇ₓf * dt # state transition matrix 
    # @printf("∇ₓf : %s\n", ∇ₓf)


    G = [zeros(3,3); Matrix{Float64}(I(3))]
    L = I + zeros(nx, nx) * dt
    # L = G * dt 
    P̂ = F * P̂_apr * F' + L * Q * L' # covariance matrix prediction
    # @printf("P̂: %s\n", P̂)
    H = get_obs_derivative(prob.h, x̂, zeros(nv)) # partial derivative of observation
    # @printf("H: %s\n", H)

    y = z - prob.h(x̂, zeros(nv)) # measurement residual
    # @printf("h: %s\n", prob.h(x̂, zeros(nv)))
    S = H * P̂ * H' + R
    K = P̂ * H' * inv(S) # Kalman gain
    x̂_new = x̂ + K * y # state estimate update
    P̂_new = (I - K * H) * P̂ # covariance estimate update
    return x̂_new, P̂_new
end

function solve_UKF()

end
