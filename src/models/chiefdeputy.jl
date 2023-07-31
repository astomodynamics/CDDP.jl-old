#################################################################################################
#=
    Model for cart pole dynamics
        from MIT's lecture on Underactuated Robotics
=#
#################################################################################################

using LinearAlgebra
using Distributions
using Random
using Symbolics

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
        r_max=1000., # maximum distance
        v_max=1., # maximum velocity
        m_max=1., # maximum mass
        T_max=1., # maximum time
        Re=6378137., # earth's radius
        μ=3.986004415e+14, # gravitational parameter around earth
        J2=1.08262668e-3, # J2 perturbation constant
        g=9.8067, # gravitational acceleration
        k_J2=3 * J2 * μ * Re^2 / 2,
        Isp=2000., 
        Œ=nothing,
        np=10,
        arr=[r_max, v_max, m_max, T_max, Re, μ, J2, g, k_J2, Isp],
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

    ∇f::Function # derivative of dynamics
    ∇²f::Function # second derivative of dynamics

    fc::Function # chief dynamic equation of motion (out-of-place)
    fc!::Function # chief dynamic equation of motion without noise (in place)

    f_cuda::Function # dynamic equation of motion (GPU, Float32)
    f_cuda!::Function # dynamic equation of motion (GPU, Float32)
    
    G::Function # noise matrix (out-of-place)
    G!::Function # noise matrix (in-place)

    h::Function # measurement model (out-of-place)

    # dynamics constants
    params::DynamicsParameter
    
    function ChiefDeputy(;)

        dims = ModelDimension(nx=6, nu=3, nw=6, ny=3, nv=6)
        
        params = DynamicsParameter()

        r_max = params.r_max
        v_max = params.v_max
        m_max = params.m_max
        T_max = params.T_max

        """ circular orbit"""
        x_init = [
            -93.89268872140511 / r_max
            68.20928216330306 / r_max
            34.10464108165153 / r_max
            0.037865035768176944 / v_max
            0.2084906865487613 / v_max
            0.10424534327438065 / v_max
            # 4.0 / m_max
        ]

        x_final =[
            -37.59664132226163 / r_max
            27.312455860666148 / r_max
            13.656227930333074 / r_max
            0.015161970413423813 / v_max
            0.08348413138390476 / v_max
            0.04174206569195238 / v_max
            # 3.9 / m_max
        ]

        œ_init = [
            6.8642335934215095e6
            1.3252107139528522
            5.233336311343717e10
            1.710422666954443
            0.17453292519943295
            0.5239464999775999
        ]

        """ elliptical orbit (Molniya)"""
        # x_init = [
        #     1000 / r_max
        #     0.  / r_max
        #     2000  / r_max
        #     0.
        #     -0.6974518910635283 / v_max
        #     0.
        #     # 10.0 / m_max
        # ]

        # x_final = [ 
        #     500 / r_max
        #     0 / r_max
        #     1000 / r_max
        #     0. 
        #     -0.34872594553176417 / v_max
        #     0.
        #     # 9.0 / m_max
        # ] 

        # œ_init = [
        #     1.5179999999999998e7
        #     1.4733342674004238e-28
        #     1.0052243705904616e11
        #     1.096066770252439
        #     0.0
        #     5.545640604761183e-32
        # ]

        """ elliptical orbit (GTO)"""
        # x_init = [
        #     100 / r_max
        #     0.  / r_max
        #     200  / r_max
        #     -0.0 / v_max
        #     -0.2295262475850926 / v_max
        #     -0.0 / v_max
        #     # 500.0 / m_max
        # ]

        # x_final = [
        #     20 / r_max
        #     0.  / r_max
        #     40  / r_max
        #     -0.0 / v_max
        #     -0.04590524951701853/ v_max
        #     -0.0 / v_max
        #     #  400/ m_max
        # ]

        # # osculating orbital elements
        # œ_init =[6.873777579835884e6, -0.002229583937249078, 6.8640542098738884e10, 0.05001634091439573, 0.00026150952573454346, 0.34880515073050467]

        """ test case 0"""
        # x_init = [
        #     1250.
        #     0.
        #     2500.
        #     0.
        #     -2.850133358359352
        #     0.
        # ]
        # x_final =  [
        #     1250.
        #     0.
        #     2500.
        #     0.
        #     -2.850133358359352
        #     0.
        # ]
        # œ_init = [6.764425942881095e6, 128.2935876689542, 5.313179817033652e10, 0.7853981633974483, 0.0, 0.0]

        """ test case 1"""
        # x_init = [
        #     500 / r_max
        #     0.  / r_max
        #     1000  / r_max
        #     -0.0 / v_max
        #     -1.03486 / v_max
        #     -0.0 / v_max
        # ]

        # x_final = [
        #     500 / r_max
        #     0.  / r_max
        #     1000  / r_max
        #     0.0
        #     -1.03564 /v_max 
        #     -1.05058/v_max
        # ]

        # œ_init = [7.199999999999999e6, 5.29819064670139e-13, 5.61864351661502e10, 1.1071478656733509, 0.0, 0.7853981633974483]
        
        """ test case 2"""
        # x_init = [
        #     500 / r_max
        #     0.  / r_max
        #     866.0254  / r_max
        #     -0.0 / v_max
        #     -1. / v_max
        #     0. / v_max
        # ]

        # x_final = [
        #     1000. / r_max
        #     0.  / r_max
        #     1732.1  / r_max
        #     0.0 / v_max
        #     2. / v_max 
        #     0 / v_max
        # ]

        # œ_init = [7.0929e6, 0.0, 5.31983111065273e10, 0.7853981633974483, 0.7853981633974483, 0.5235987755982988]
        
        # dynamic equation of motion
        @variables t du[1:dims.nx] u[1:dims.nx] p[1:params.np+dims.nx+dims.nu] real=true
        du = collect(du)
        symbolic_dynamics!(du, u, p, t)
        f_base! = build_function(du, u, p, t, target=Symbolics.CTarget(), expression=Val{false})
        
        f!(dx, x, params, t) = begin
            p = get_ode_input(x, params, t)
            f_base!(dx, x, p, t)
        end

        f(x, params, t) = begin
            dx = zeros(dims.nx)
            p = get_ode_input(x, params, t)
            f_base!(dx, x, p, t)
            return dx
        end

        # derivative of dynamics
        symbolic_∇ₓf! = Symbolics.jacobian(du, u)
        symbolic_∇ᵤf! = Symbolics.jacobian(du, p[params.np+1+dims.nx:params.np+dims.nx+dims.nu])
        symbolic_∇ₓₓf! = Symbolics.jacobian(symbolic_∇ₓf!, u)
        symbolic_∇ₓᵤf! = Symbolics.jacobian(symbolic_∇ₓf!, p[params.np+1+dims.nx:params.np+dims.nx+dims.nu])
        symbolic_∇ᵤᵤf! = Symbolics.jacobian(symbolic_∇ᵤf!, p[params.np+1+dims.nx:params.np+dims.nx+dims.nu])
        ∇ₓf_base! = build_function(symbolic_∇ₓf!, u, p, t, target=Symbolics.CTarget(), expression=Val{false})
        ∇ᵤf_base! = build_function(symbolic_∇ᵤf!, u, p, t, target=Symbolics.CTarget(), expression=Val{false})
        ∇ₓₓf_base! = build_function(symbolic_∇ₓₓf!, u, p, t, target=Symbolics.CTarget(), expression=Val{false})
        ∇ₓᵤf_base! = build_function(symbolic_∇ₓᵤf!, u, p, t, target=Symbolics.CTarget(), expression=Val{false})
        ∇ᵤᵤf_base! = build_function(symbolic_∇ᵤᵤf!, u, p, t, target=Symbolics.CTarget(), expression=Val{false})
        
        ∇f(x, params) = begin
            t = 0.
            p = get_ode_input(x, params, t)
            nx = size(x, 1)
            nu = size(params.U_ref, 1)
            ∇ₓf = zeros(nx, nx)
            ∇ᵤf = zeros(nx, nu)
            ∇ₓf_base!(∇ₓf, x, p, t)
            ∇ᵤf_base!(∇ᵤf, x, p, t)
            return ∇ₓf, ∇ᵤf
        end

        ∇²f(x, params) = begin
            t = 0.
            p = get_ode_input(x, params, t)
            nx = size(x, 1)
            nu = size(params.U_ref, 1)
            ∇ₓₓf = zeros(nx, nx, nx)
            ∇ₓᵤf = zeros(nx, nx, nu)
            ∇ᵤᵤf = zeros(nx, nu, nu)
            ∇ₓₓf_base!(∇ₓₓf, x, p, t)
            ∇ₓᵤf_base!(∇ₓᵤf, x, p, t)
            ∇ᵤᵤf_base!(∇ᵤᵤf, x, p, t)
            return ∇ₓₓf, ∇ₓᵤf, ∇ᵤᵤf
        end

        
        new(
            dims,
            x_init,
            x_final,
            œ_init,
            f,
            f!,
            ∇f,
            ∇²f,
            fc,
            fc!,
            f_cuda,
            f_cuda!,
            empty,
            empty,
            h,
            params
        )
    end
end


"""
    symbolic_dynamics!(dx, x, p, t)

The dynamic equation of motion.

# Arguments
- `dx`: derivative of state at a given time step
- `x`: state at a given time step
- `p`: parameter arguments
- `t`: time
"""
function symbolic_dynamics!(dx, x, p, t)
    # unpack parameters
    (r_max, v_max, m_max, T_max,
        Re, μ, J2, g, k_J2, Isp, r, vx, h, i, Ω, θ, ux, uy, uz) = p

    xj = copy(x[1]) * r_max
    yj = copy(x[2]) * r_max
    zj = copy(x[3]) * r_max
    ẋj = copy(x[4]) * v_max
    ẏj = copy(x[5]) * v_max
    żj = copy(x[6]) * v_max
    # mj = copy(x[7]) * m_max
    
    # precompute constants from chief dynamics
    n² = μ / r^3 + k_J2 / r^5 - 5 * k_J2 * sin(i)^2 * sin(θ)^2 / r^5
    ζ = 2 * k_J2 * sin(i) * sin(θ) / r^4
    ωx = -k_J2 * sin(2 * i) * sin(θ) / (h * r^3) 
    ωz = h / r^2
    ωy = 0
    ω̇x = -k_J2 * sin(2 * i) * cos(θ) / r^5  + 3 * vx * k_J2 * sin(2 * i) * sin(θ) / (r^4 * h) -
        8 * k_J2^2 * sin(i)^3 * cos(i) * sin(θ)^2 * cos(θ) / (r^6 * h^2)
    ω̇z = -2 * h * vx / r^3 - k_J2 * sin(i)^2 * sin(2 * θ) / r^5

    # deputy Dynamics ẋ
    rj = sqrt((r + xj)^2 + yj^2 + zj^2)
    rjz = (r + xj) * sin(i) * sin(θ) + yj * sin(i) * cos(θ) + zj * cos(i)
    ζj = 2 * k_J2 * rjz / rj^5
    n²j = μ / rj^3 + k_J2 / rj^5 - 5 * k_J2 * rjz^2 / rj^7

    ẍ = 2 * ẏj * ωz - xj * (n²j - ωz^2) + yj * ω̇z - zj * ωx * ωz - (ζj - ζ) * sin(i) * sin(θ) - 
        r * (n²j - n²) 
    ÿ = -2 * ẋj * ωz + 2 * żj * ωx - xj * ω̇z - yj * (n²j - ωz^2 - ωx^2) + zj * ω̇x - 
        (ζj - ζ) * sin(i) * cos(θ)
    z̈ = -2 * ẏj * ωx - xj * ωx * ωz - yj * ω̇x - zj * (n²j - ωx^2) - 
        (ζj - ζ) * cos(i) 
    mj = 1
    # nonlinear equations of motion 
    dx[1:6] = T_max * [
        ẋj / r_max
        ẏj / r_max
        żj / r_max 
        ẍ / v_max + ux / mj 
        ÿ / v_max + uy / mj
        z̈ / v_max + uz / mj 
        # 0
    ]
end

function f_cuda(x, p, t)
    """ dynamic equation of motion (GPU, Float32)"""
    dx = zeros(length(x))
    f_cuda!(dx, x, p, t)
    return dx
end

function f_cuda!(dx, x, p, t)
    """ dynamic equation of motion (GPU, Float32)"""
    # unpack parameters
    (r_max, v_max, m_max, T_max,
        Re, μ, J2, g, k_J2, Isp, r, vx, h, i, Ω, θ, ux, uy, uz) = p
    
    xj = CUDA.copy(x[1]) * r_max
    yj = CUDA.copy(x[2]) * r_max
    zj = CUDA.copy(x[3]) * r_max
    ẋj = CUDA.copy(x[4]) * v_max
    ẏj = CUDA.copy(x[5]) * v_max
    żj = CUDA.copy(x[6]) * v_max

    # precompute constants from chief dynamics
    n² = μ / r^3 + k_J2 / r^5 - 5 * k_J2 * CUDA.sin(i)^2 * CUDA.sin(θ)^2 / r^5
    ζ = 2 * k_J2 * CUDA.sin(i) * CUDA.sin(θ) / r^4
    ωx = -k_J2 * CUDA.sin(2 * i) * CUDA.sin(θ) / (h * r^3) 
    ωz = h / r^2
    ωy = 0
    ω̇x = -k_J2 * CUDA.sin(2 * i) * CUDA.cos(θ) / r^5  + 3 * vx * k_J2 * CUDA.sin(2 * i) * CUDA.sin(θ) / (r^4 * h) -
        8 * k_J2^2 * CUDA.sin(i)^3 * CUDA.cos(i) * CUDA.sin(θ)^2 * CUDA.cos(θ) / (r^6 * h^2)
    ω̇z = -2 * h * vx / r^3 - k_J2 * CUDA.sin(i)^2 * CUDA.sin(2 * θ) / r^5

    # deputy Dynamics ẋ
    rj = CUDA.sqrt((r + xj)^2 + yj^2 + zj^2)
    rjz = (r + xj) * CUDA.sin(i) * CUDA.sin(θ) + yj * CUDA.sin(i) * CUDA.cos(θ) + zj * CUDA.cos(i)
    ζj = 2 * k_J2 * rjz / rj^5
    n²j = μ / rj^3 + k_J2 / rj^5 - 5 * k_J2 * rjz^2 / rj^7

    ẍ = 2 * ẏj * ωz - xj * (n²j - ωz^2) + yj * ω̇z - zj * ωx * ωz - (ζj - ζ) * CUDA.sin(i) * CUDA.sin(θ) - 
        r * (n²j - n²) 
    ÿ = -2 * ẋj * ωz + 2 * żj * ωx - xj * ω̇z - yj * (n²j - ωz^2 - ωx^2) + zj * ω̇x - 
        (ζj - ζ) * CUDA.sin(i) * CUDA.cos(θ)
    z̈ = -2 * ẏj * ωx - xj * ωx * ωz - yj * ω̇x - zj * (n²j - ωx^2) - 
        (ζj - ζ) * CUDA.cos(i) 

    # nonlinear equations of motion 
    dx[1:6] = T_max * [
        ẋj / r_max
        ẏj / r_max
        żj / r_max 
        ẍ / v_max + ux / mj 
        ÿ / v_max + uy / mj
        z̈ / v_max + uz / mj 
    ]
end

"""
    fc(x, p, t)
"""

function fc(x, p, t)
    dx = zeros(length(x))
    fc!(dx, x, p, t)
    return dx
end 

"""
    fc!(dx, x, p, t)
"""

function fc!(dx::Vector, œ::Vector, p::ODEParameter, t::Float64)
    # unpack parameters
    params = get_ode_input(œ, p, t)
    (r_max, v_max, m_max, T_max,
        Re, μ, J2, g, k_J2, Isp, _ , u) = params
    
    r = copy(œ[1]) # radius
    vx = copy(œ[2]) # radial velocity
    h = copy(œ[3]) # angular momentum
    i = (copy(œ[4])) # inclinaiton
    Ω = (copy(œ[5])) # RAAN
    θ = (copy(œ[6])) # argument of latitude, theta

    # chief dynamics æ̇; æ̇_dim = 6
    dx[1:6] = T_max * [
        vx
        -μ / r^2 + h^2 / r^3 - k_J2 * (1 - 3 * sin(i)^2 * sin(θ)^2) / r^4
        -k_J2 * sin(i)^2 * sin(2 * θ) / r^3 
        -k_J2 * sin(2 * i) * sin(2 * θ) / (2 * h * r^3) 
        -2 * k_J2 * cos(i) * sin(θ)^2 / (h * r^3) 
        h / r^2 + 2 * k_J2 * cos(i)^2 * sin(θ)^2 / (h * r^3)
    ]
    return dx
end


function get_ode_input(x, p, t)
    U = p.U_ref
    X_ref = p.X_ref
    U_md = p.U_md
    u = nothing
    œ = nothing

    if isnothing(p.U_ref)
        u = nothing
    elseif isa(p.U_ref, Vector)
        # check if the reference control is array or function
        u = U
    else
        u = U(t)
    end

    if isnothing(U_md)
        œ = zeros(6)
    elseif isa(U_md, Vector)
        œ = U_md
    else
        œ = U_md(t)
    end
    return [p.params; œ; u]
end

function h(x::Vector, v::Vector)

    # y = [
    #     norm(x[1:3])
    #     atan(x[1]/x[2])
    #     atan(x[3]/norm(x[1:2]))
    # ]
    y = I * x
    return y
end

