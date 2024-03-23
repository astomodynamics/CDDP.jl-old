module CDDPOld

using LinearAlgebra
using ForwardDiff
using Distributions
using Random
using Printf

using DifferentialEquations
using DataInterpolations

# API
export # type
    AbstractDynamicsModel,
    AbstractObservationModel,
    AbstractDDPProblem,
    AbstractDDOFunction,
    AbstractDDPArray,
    AbstractDDPSolution,
    AbstractDDPParameter,
    AbstractMPPIProblem,
    AbstractMPPIParameter,
    AbstractMPPIFunction,
    AbstractMPPISolution
include("./type_definition.jl")


# helper functions
export 
    ODEParameter,
    ModelDimension,
    CostFunction,
    DynamicsFunction,
    MPPICostFunction,
    MPPIDynamicsFunction,
    ConstraintFunction,
    get_dims,
    get_ode_input,
    initialize_trajectory,
    simulate_trajectory,
    get_ode_derivatives,
    get_obs_derivative,
    get_instant_cost_derivatives,
    get_terminal_cost_derivatives,
    get_instant_const_derivative,
    get_obs_derivatives,
    rk4_step,
    rk4_step_cuda,
    rk2_step,
    rk2_step_cuda,
    euler_step,
    euler_step_cuda,
    get_continuous_dynamics,
    get_discrete_dynamics,
    get_feasibility,
    get_trajectory_cost,
    get_trajectory_log_cost
include("./helper.jl")

# ddp problems
export 
    DDPProblem,
    CDDPProblem,
    DDPGain
include("./ddp_problem.jl")

# ddp solvers
export 
    solve_ddp,
    solve_cddp
include("./ddp_solver.jl")


# backward pass functions
export
    backward_pass_ddp!,
    backward_pass_cddp!
include("./backward_pass.jl")

# forward pass functions
export 
    forward_pass_ddp!,
    forward_pass_cddp!
include("./forward_pass.jl")

# visualization
# export
#     make_gif
# include("./animate.jl")

# mppi solvers
export
    MPPIProblemCPU,
    MPPIProblemGPU,
    solve_mppi_cpu,
    solve_mppi_gpu,
    solve_mppi_old,
    simulate_mppi

include("./controller.jl")

# filters
export 
    KFProblem,
    solve_KF,
    EKFProblem,
    solve_EKF
include("./filter.jl")

# export 
#     plot_arrow2d!,
#     plot_arrow3d!
# include("./visualize.jl")

end # module CDDP