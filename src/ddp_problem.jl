################################################################################
#=
    DDP solver environment

    This DDP environment is for standard DDP, Constrained DDP (CDDP)
=#
################################################################################


struct iLQRProblem <: AbstractDDPProblem
    model::AbstractDynamicsModel

    # problem setting
    tf::Float64 # final time 
    tN::Int64 # number of discretization steps
    dt::Float64 # discretization step-size

    # dimensions
    x_dim::Int64 # state dimension
    u_dim::Int64 # control dimension

    # cost objective
    ell::Function # instantaneous cost function (running cost function)
    ϕ::Function # terminal cost function

    # dynamics
    f::Function # dynamics model function
    ∇f::Function # derivative of dynamics

    # boundary conditions
    x_init::Vector{Float64} # initial state 
    x_final::Vector{Float64} # terminal state

    X_ref # reference trajectory
    function iLQRProblem(;
        model::AbstractDynamicsModel,
        tf::Float64=model.tf,
        tN::Int64=model.tN,
        dt::Float64=model.dt,
        x_dim = model.x_dim,
        u_dim = model.u_dim,
        ell::Function=model.ell,
        ϕ::Function=model.ϕ,
        f::Function=model.f,
        ∇f::Function=model.∇f,
        x_init::Vector{Float64}=model.x_init,
        x_final::Vector{Float64}=model.x_final,
        X_ref=nothing,
        ) 
        new(
            model,
            tf,
            tN,
            dt,
            x_dim,
            u_dim,
            ell,
            ϕ,
            f,
            ∇f,
            x_init,
            x_final,
            X_ref
        )
    end
end

struct DDPProblem <: AbstractDDPProblem
    model::AbstractDynamicsModel

    # problem setting
    tf::Float64 # final time 
    tN::Int64 # number of discretization steps
    dt::Float64 # discretization step-size

    # dimensions
    x_dim::Int64 # state dimension
    u_dim::Int64 # control dimension

    # cost objective
    ell::Function # instantaneous cost function (running cost function)
    ϕ::Function # terminal cost function

    # dynamics
    f::Function # dynamics model function
    ∇f::Function # derivative of dynamics

    # boundary conditions
    x_init::Vector{Float64} # initial state 
    x_final::Vector{Float64} # terminal state

    X_ref # reference trajectory
    function DDPProblem(;
        model::AbstractDynamicsModel=model,
        tf::Float64=model.tf,
        tN::Int64=model.tN,
        dt::Float64=model.dt,
        x_dim = model.x_dim,
        u_dim = model.u_dim,
        ell::Function=model.ell,
        ϕ::Function=model.ϕ,
        f::Function=model.f,
        ∇f::Function=model.∇f,
        x_init::Vector{Float64}=model.x_init,
        x_final::Vector{Float64}=model.x_final,
        X_ref=nothing,
        ) 
        new(
            model,
            tf,
            tN,
            dt,
            x_dim,
            u_dim,
            ell,
            ϕ,
            f,
            ∇f,
            x_init,
            x_final,
            X_ref
        )
    end
end

struct CDDPProblem <: AbstractDDPProblem
    model::AbstractDynamicsModel

    # problem setting
    tf::Float64 # final time 
    tN::Int64 # number of discretization steps
    dt::Float64 # discretization step-size

    # dimensions
    x_dim::Int64 # state dimension
    u_dim::Int64 # control dimension
    λ_dim::Int64 # constraint dimension

    # cost objective
    ell::Function # instantaneous cost function (running cost function)
    ϕ::Function # terminal cost function

    # dynamics
    f::Function # dynamics model function
    ∇f::Function # derivative of dynamics

    # boundary conditions
    x_init::Vector{Float64} # initial state 
    x_final::Vector{Float64} # terminal state

    # constraints
    c::Function # instantaneous constraint function (running cost function)
    c_final::Function # termianl constraint function

    X_ref # reference trajectory
    function CDDPProblem(;
        model::AbstractDynamicsModel,
        tf::Float64=model.tf,
        tN::Int64=model.tN,
        dt::Float64=model.dt,
        x_dim = model.x_dim,
        u_dim = model.u_dim,
        λ_dim = model.λ_dim,
        ell::Function=model.ell,
        ϕ::Function=model.ϕ,
        f::Function=model.f,
        ∇f::Function=model.∇f,
        x_init::Vector{Float64}=model.x_init,
        x_final::Vector{Float64}=model.x_final,
        c::Function=model.c,
        c_final::Function=model.c_final,  
        X_ref=nothing,
        ) 
        new(
            model,
            tf,
            tN,
            dt,
            x_dim,
            u_dim,
            λ_dim,
            ell,
            ϕ,
            f,
            ∇f,
            x_init,
            x_final,
            c,
            c_final,
            X_ref
        )
    end
end


mutable struct DDPSolution <: DDPSolutions
    X # X trajectory storage
    U # U trajectory storage
    J::Float64 # cost storage
    gains::DDPArrays
end

mutable struct CDDPSolution <: DDPSolutions
    X # X trajectory storage
    U # U trajectory storage
    Λ # λ trajectory storage
    Y # y trajectory storage
    J::Float64 # cost storage
    gains::DDPArrays
end

mutable struct DDPGain <: DDPArrays
    l # feedforward gain for x
    L # feedback gain for x
end

mutable struct CDDPGain <: DDPArrays
    l # feedforward gain for x
    L # feedback gain for x
    m # coefficients for y
    M # coefficients for y
    n # coefficients for y
    N # coefficients for y
end
