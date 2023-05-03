module CDDP

using LinearAlgebra
using Plots; gr()
using ForwardDiff
using Distributions
using Random
using Printf

using DifferentialEquations
using Interpolations

# API
export # type
    AbstractDynamicsModel,
    AbstractObservationModel,
    AbstractDDPProblem,
    DDPArrays,
    DDPSolutions,
    Parameters
include("./type_definition.jl")


# ddp problems
export 
    iLQRProblem,
    DDPProblem,
    CDDPProblem,
    DDPGain
include("./ddp_problem.jl")

# ddp solvers
export 
    solve_ilqr,
    solve_ddp,
    solve_cddp
include("./ddp_solver.jl")


# backward pass functions
export
    backward_pass_ilqr!,
    backward_pass_ddp!,
    backward_pass_cddp!
include("./backward_pass.jl")

# forward pass functions
export 
    forward_pass_ilqr!,
    forward_pass_ddp!,
    forward_pass_cddp!
include("./forward_pass.jl")

# helper functions
export 
    ODEParams,
    initialize_trajectory,
    simulate_trajectory,
    get_ode_derivatives,
    get_instant_cost_derivatives,
    get_terminal_cost_derivatives,
    get_instant_const_derivative,
    get_obs_derivatives,
    rk4_step,
    rk2_step,
    euler_step,
    get_feasibility,
    get_trajectory_cost,
    get_trajectory_log_cost
include("./helper.jl")

end # module CDDP