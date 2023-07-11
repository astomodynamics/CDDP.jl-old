#################################################################################################
#=
    Model for cart pole dynamics
        from MIT's lecture on Underactuated Robotics
=#
#################################################################################################

using LinearAlgebra
using Distributions
using Random

export ChiefDeputy

mutable struct DynamicsParameter
    r_max::Float64
    v_max::Float64
    m_max::Float64
    T_max::Float64
    Re::Float64
    μ::Float64
    J2::Float64
    g::Float64
    k_J2::Float64
    Isp::Float64
    Œ

    np::Int64 # number of parameters
    arr::Vector{Float64} # parameter array

    function DynamicsParameter(;
        r_max=200, # maximum distance
        v_max=1, # maximum velocity
        m_max=10, # maximum mass
        T_max=1, # maximum time
        Re=6378137, # earth's radius
        μ=3.986004415 * 10^14, # gravitational parameter around earth
        J2=1.08262668*10^-3, # J2 perturbation constant
        g=9.8067, # gravitational acceleration
        k_J2=3 * J2 * μ * Re^2 / 2,
        Isp=1500, # MIT electrospray spec
        Œ=nothing,
        np=11,
        arr=[r_max, v_max, m_max, T_max, Re, μ, J2, g, k_J2, Isp, 0],
    )
        new(
            r_max,
            v_max,
            m_max,
            T_max,
            Re, 
            μ, 
            J2, g, 
            k_J2, 
            Isp, 
            Œ,
            np,
            arr
        )
    end
end

# ChiefDeputy as a Julia class
mutable struct ChiefDeputy<: AbstractDynamicsModel
    dims::ModelDimension # dimension of the state, control, constraint, and noise

    # Boundary  conditions
    x_init::Vector{Float64}
    x_final::Vector{Float64}
    œ_init::Vector{Float64}

    # function storage
    f::Function # dynamic equation of motion (out-of-place)
    f!::Function # dynamic equation of motion without noise (in place)

    fc::Function # chief dynamic equation of motion (out-of-place)
    fc!::Function # chief dynamic equation of motion without noise (in place)
    
    G::Function # noise matrix (out-of-place)
    G!::Function # noise matrix (in-place)

    h::Function # measurement model (out-of-place)

    # dynamics constants
    params::DynamicsParameter
    
    function ChiefDeputy(;)

        dims = ModelDimension(nx=6, nu=3)
        
        params = DynamicsParameter()

        r_max = params.r_max
        v_max = params.v_max
        m_max = params.m_max
        T_max = params.T_max

        x_init = [
            -93.89268872140511 / r_max
            68.20928216330306 / r_max
            34.10464108165153 / r_max
            0.037865035768176944 / v_max
            0.2084906865487613 / v_max
            0.10424534327438065 / v_max
            4.0 / m_max
        ]

        x_final =[
            -37.59664132226163 / r_max
            27.312455860666148 / r_max
            13.656227930333074 / r_max
            0.015161970413423813 / v_max
            0.08348413138390476 / v_max
            0.04174206569195238 / v_max
            3.9 / m_max
        ]

        œ_init = [
            6.8642335934215095e6
            1.3252107139528522
            5.233336311343717e10
            1.710422666954443
            0.17453292519943295
            0.5239464999775999
        ]

        


        new(
            dims,
            x_init,
            x_final,
            œ_init,
            f,
            f!,
            fc,
            fc!,
            empty,
            empty,
            empty,
            params
        )
    end
end

function f(x, p, t)
    dx = zeros(6)
    f!(dx, x, p, t)
    return dx
end 

"""
    f(x, p, t)

The dynamic equation of motion.

# Arguments
- `x`: state at a given time step
- `p`: parameter arguments
- `t`: time
"""
function f!(dx::Vector, x::Vector, p::ODEParameter, t::Float64)

    # unpack parameters
    params = get_ode_input(p, t)
    (r_max, v_max, m_max, T_max,
        Re, μ, J2, g, k_J2, Isp, œ , u) = params

    r = copy(œ[1]) # radius
    vx = copy(œ[2]) # radial velocity
    h = copy(œ[3]) # angular momentum
    i = copy(œ[4]) # inclinaiton
    Ω = copy(œ[5]) # RAAN
    θ = copy(œ[6]) # argument of latitude, theta

    xj = copy(x[1]) * r_max
    yj = copy(x[2]) * r_max
    zj = copy(x[3]) * r_max
    ẋj = copy(x[4]) * v_max
    ẏj = copy(x[5]) * v_max
    żj = copy(x[6]) * v_max
    mj = copy(x[7]) * m_max
    
    η² = μ / r^3 + k_J2 / r^5 - 5 * k_J2 * sin(i)^2 * sin(θ)^2 / r^5
    ζ = 2 * k_J2 * sin(i) * sin(θ) / r^4
    ω̇x = -k_J2 * sin(2 * i) * cos(θ) / r^5 
    ω̇z = -2 * h * vx / r^3 - k_J2 * sin(i)^2 * sin(2 * θ) / r^5
    ωx = -k_J2 * sin(2 * i) * sin(θ) / (h * r^3) 
    ωz = h / r^2
    ωy = 0

    # deputy Dynamics ẋ
    rj = sqrt((r + xj)^2 + yj^2 + zj^2)
    rjz = (r + xj) * sin(i) * sin(θ) + yj * sin(i) * cos(θ) + zj * cos(i)
    ζj = 2 * k_J2 * rjz / rj^5
    η²j = μ / rj^3 + k_J2 / rj^5 - 5 * k_J2 * rjz^2 / rj^7

    v̇1 = 2 * ẏj * ωz - xj * (η²j - ωz^2) + yj * ω̇z - zj * ωx * ωz - (ζj - ζ) * sin(i) * sin(θ) - 
        r * (η²j - η²) 
    v̇2 = -2 * ẋj * ωz + 2 * żj * ωx - xj * ω̇z - yj * (η²j - ωz^2 - ωx^2) + zj * ω̇x - 
        (ζj - ζ) * sin(i) * cos(θ)
    v̇3 = -2 * ẏj * ωx - xj * ωx * ωz - yj * d_max * ω̇x - zj * (η²j - ωx^2) - 
        (ζj - ζ) * cos(i) 
    
    # nonlinear equations of motion 
    dx[1:6] = T_max * [
        ẋj / d_max
        ẏj / d_max
        żj / d_max 
        v̇1 / v_max + u[1] / mj 
        v̇2 / v_max + u[2] / mj
        v̇3 / v_max + u[3] / mj 
        -norm(u)/(g₀*Isp) / m_max
    ]
end

function fc(x, p, t)
    dx = zeros(6)
    fc!(dx, x, p, t)
    return dx
end 

"""
    fc(x, p, t)
"""

function fc!(dx::Vector, œ::Vector, p::ODEParameter, t::Float64)
    # unpack parameters
    params = get_ode_input(œ, p, t)
    (r_max, v_max, m_max, T_max,
        Re, μ, J2, g, k_J2, Isp, _ , u) = params

    r = copy(œ[1]) # radius
    vx = copy(œ[2]) # radial velocity
    h = copy(œ[3]) # angular momentum
    i = copy(œ[4]) # inclinaiton
    Ω = copy(œ[5]) # RAAN
    θ = copy(œ[6]) # argument of latitude, theta

    # chief dynamics æ̇; æ̇_dim = 6
    dx[1:6] = T_max * [
        œ[2]
        -μ / r^2 + h^2 / r^3 - k_J2 / r^4 * (1 - 3 * sin(i)^2 * sin(θ)^2)
        -k_J2 * sin(i)^2 * sin(2 * θ) / r^3 
        -k_J2 * sin(2 * i) * sin(2 * θ) / (2 * h * r^3) 
        -2 * k_J2 * cos(i) * sin(θ)^2 / (h * r^3) 
        h / r^2 + 2 * k_J2 * cos(i)^2 * sin(θ)^2 / (h * r^3)
    ]
end


function get_ode_input(x, p, t)
    U = p.U_ref
    X_ref = p.X_ref
    u = nothing

    # check if the reference control is array or function
    if isa(U, Vector)
        u = U
    else
        # u = p.U_ref[trunc(Int, t/model.dt)+1]
        u = U(t)
    end

    if !isequal(X_ref, nothing)
        œ = X_ref(t)
    else
        œ = zeros(6)
    end

    return [p.params; œ; u]
end