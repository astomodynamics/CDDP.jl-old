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
    f!::Function # dynamics model function

    # boundary conditions
    x_init::Vector{Float64} # initial state 
    x_final::Vector{Float64} # terminal state

    X_ref # reference trajectory
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
    f!::Function # dynamics model function

    # boundary conditions
    x_init::Vector{Float64} # initial state 
    x_final::Vector{Float64} # terminal state

    X_ref # reference trajectory
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
    f!::Function # dynamics model function

    # boundary conditions
    x_init::Vector{Float64} # initial state 
    x_final::Vector{Float64} # terminal state

    # constraints
    c::Function # instantaneous constraint function (running cost function)
    c_final::Function # termianl constraint function

    X_ref # reference trajectory
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
