# ============================================================
#=
    Astrodynamics functions
=#
# ============================================================

function get_initial_y_velocity(x0, n, e)
    return -n*(2+e)*x0/(sqrt(1 + e)*sqrt(1 - e)^3)
end

"""
    Compute eccectric anomaly using Kepler's equation and the Newton-Raphson method
"""
function Kepler_eqn(E0,M,e)
        E = E0
        tol = 1e-10
        delta = 1e-10
        MaxIte = 1000
        ite = 0

        while true
            df = (((E + delta) - e*sin(E + delta) - M) -
                  ((E - delta) - e*sin(E - delta) - M))/(2*delta)
            if abs(df)<=0
                break
            end
            E_ = E # set old E
            E = E_ - (E- e*sin(E) - M)/df
            ite = ite + 1
            if abs(E - E_)<=tol || MaxIte <= ite
                break
            end
        end
        return E
    end

"""
    compute rotation matrix in dir axis from body to inertial frame
"""
function PQW2ECI(phi,dir)
    mat = zeros(3,3)
    if dir == 1
        mat = [1   0   0;
               0  cos(phi)   -sin(phi);
               0  sin(phi)   cos(phi)]
    end

    if dir == 2
        mat = [cos(phi)   0  sin(phi);
                0  1  0;
                -sin(phi)   0  cos(phi)]
    end

    if dir == 3
        mat =[cos(phi)   -sin(phi)   0;
              sin(phi)   cos(phi)  0;
              0   0   1]
    end
    return mat
end


"""

"""
function MOE2OE(æ::Vector;iseccentric=true)
    """Compute inertial frame position and velocity vectors"""
    if iseccentric
        μ = 3.986004415e+14

        a = copy(æ[1]) # a: Semi-major axis, m
        e = copy(æ[2]) # eccentricity
        i = copy(æ[3]) # inclination
        Ω = copy(æ[4]) # RAAN
        ω = copy(æ[5]) # argument of periapsis
        M = copy(æ[6]) # mean anomaly

        p = a*(1.0 - e^2) # the parameter

        E = Kepler_eqn(M+e/2.0,M,e) # eccentric anomaly
        f = 2.0*atan(sqrt((1 + e)/(1 - e))*tan(E/2)) # eccentric anomaly, rad
        r = p / (1 + e*cos(f)) # radius magnitude

        # perifocal frame position and velocity vecvor
        r_pqw = r*[cos(f);sin(f); 0]
        v_pqw = sqrt(μ/p)*[-sin(f);e+cos(f); 0]

        # set Euler rotation matrix (3-1-3)
        PQW2ECI313 = PQW2ECI(Ω,3)*PQW2ECI(i,1)*PQW2ECI(ω,3)

        # inertial frame position and velocity vector
        r_ECI = PQW2ECI313*r_pqw
        v_ECI = PQW2ECI313*v_pqw

        x_eci = [r_ECI; v_ECI]

        r = norm(r_ECI)
        v = norm(v_ECI)
        hv = cross(x_eci[1:3],x_eci[4:6]) # angular momentum vector
        h = norm(hv)
        v_x = dot(x_eci[1:3],x_eci[4:6])/norm(x_eci[1:3])
        θ = f + ω
    end
    return [r; v_x; h; i; Ω; θ]
end



function MOE2ECI(æ::Vector)
    """
    Compute inertial frame position and velocity vectors

    Args:
        alp: orbital elements using D'Amico's description

    Returns:
        x: position and velocity vector
    """
    μ = 3.986004415*10^14

    a = copy(æ[1]) # a: Semi-major axis, m
    e = copy(æ[2])
    i = copy(æ[3]) #
    Ω = copy(æ[4]) #
    ω = copy(æ[5]) #
    M = copy(æ[6]) #

    p = a*(1.0 - e^2) # the parameter

    E = Kepler_eqn(M+e/2.0,M,e) # eccentric anomaly
    f = 2.0*atan(sqrt((1 + e)/(1 - e))*tan(E/2)) # eccentric anomaly, rad
    r = p / (1 + e*cos(f)) # radius magnitude

    # perifocal frame position and velocity vecvor
    r_pqw = r*[cos(f); sin(f); 0]
    v_pqw = sqrt(μ/p)*[-sin(f);e+cos(f); 0]

    # set Euler rotation matrix (3-1-3)
    PQW2ECI313 = PQW2ECI(Ω,3)*PQW2ECI(i,1)*PQW2ECI(ω,3)

    # inertial frame position and velocity vector
    r_ECI = PQW2ECI313*r_pqw
    v_ECI = PQW2ECI313*v_pqw

    x = [r_ECI; v_ECI]
    return x
end



function ECI2LVLH(x_c::Vector,rho::Vector)
    """
    compute position and velocity vector in rtn frame
    from inertial frame position and velocity vector
    """
    r = x_c[1:3]
    h = cross(x_c[1:3],x_c[4:6])
    ω = cross(x_c[1:3],x_c[4:6])/norm(x_c[1:3])^2
    x̂ = r/norm(r)
    ẑ = h/norm(h)
    ŷ = cross(ẑ,x̂)

    x̂̇ = cross(ω,x̂)
    ŷ̇ = cross(ω,ŷ)
    ẑ̇ = cross(ω,ẑ)

    A_IO = [x̂   ŷ   ẑ]
    B_IO = [x̂̇   ŷ̇   ẑ̇]
    r_d_RTN = A_IO'*rho[1:3]
    v_d_RTN = A_IO'*rho[4:6]
    return [r_d_RTN; v_d_RTN]
end


function ECI2RTN(xc::Vector, ρ::Vector)
    nothing
end

function MOE2ROE(æc,æd)
    """
    compute relative orbital elements description from chief and deputy orbital elements
    """
    ac = copy(æc[1])
    ec = copy(æc[2])
    ic = copy(æc[3])
    Ωc = copy(æc[4])
    ωc = copy(æc[5])
    Mc = copy(æc[6])
    ad = copy(æd[1])
    ed = copy(æd[2])
    id = copy(æd[3])
    Ωd = copy(æd[4])
    ωd = copy(æd[5])
    Md = copy(æd[6])

    δα = [(ad - ac)/ac;
          Md - Mc + ωd - ωc + (Ωd - Ωc)*cos(ic);
          ed*cos(ωd) - ec*cos(ωc);
          ed*sin(ωd) - ec*sin(ωc);
          id - ic;
          (Ωd - Ωc)*sin(ic)]
    return δα
end

function ECI2MOE(x::Vector)
    μ = 3.986004415*10^14
    r = norm(x[1:3]) # radius (m)
    v = norm(x[4:6]) # velocity (m/s)
    hv = cross(x[1:3],x[4:6]) # angular momentum vector
    n = cross([0; 0; 1],hv/norm(hv)) # nodal vector

    a = r/(2.0 - r*v^2/μ) # a: semimajor-axis
    ev = ((v^2 - μ/r)*x[1:3] - x[1:3]'*x[4:6]*x[4:6])/μ # eccentricity vector
    e = norm(ev) # eccentricity
    i = acos(dot(hv,[0, 0, 1])/norm(hv)) # inclination, rad
    W = acos(dot(n,[1, 0, 0])/norm(n)) # Right ascersion right ascending node, rad
    if dot(n,[0,1,0])<0
        W = 2*pi - W
    end

    cos_w = dot(n,ev)/(norm(n)*norm(ev))
    w = acos(cos_w) # argument of periapsis, rad
    if dot(ev,[0,0,1]) < 0
        w = 2*pi - w
    end
    f = acos(dot(ev,x[1:3])/norm(ev)/norm(x[1:3])) # true anomaly, rad
    if dot(x[1:3],x[4:6]) < 0
        f = 2*np.pi - f
    end

    E = 2.0*atan(sqrt((1 - e)/(1 + e))*tan(f/2)) # eccentric anomaly, rad
    M = E - e*sin(E) # mean anomaly, rad

    return [a, e, i, W, w, M]
end

function ROE2MOE(æ,δα)
    """
    Compute deputy orbital elements from the chief orbital elements and the relative orbital elements
    This can be singular if the inclination is 90 deg
    """

    a = copy(æ[1])
    e = copy(æ[2])
    i = copy(æ[3])
    Ω = copy(æ[4])
    ω = copy(æ[5])
    M = copy(æ[6])

    αc = [a; M+ω; e*cos(ω); e*sin(ω); i; Ω]

    αd = [δα[1]*αc[1] + a;
          δα[2] + αc[2] - δα[6]/tan(αc[5]);
          δα[3] + αc[3];
          δα[4] + αc[4];
          δα[5] + αc[5];
          δα[6]/sin(αc[5])+αc[6]]

    ad = αd[1] # a: Semi-major axis, m
    ud = αd[2] # the mean argument of latitude, rad
    exd = αd[3] # eccentricity vector component of x
    eyd = αd[4] # eccentricity vector component of y
    id = αd[5] # i: inclination, rad
    Ωd = αd[6] # Omega: Right ascension of the ascending node, RAAN, rad)
    ed = sqrt(exd^2 + eyd^2) #  eccentricity

    ωd = 0
    # argument of periapsis, w (rad)
    if exd < 0 && eyd > 0
        ωd = atan(eyd/exd) + pi
    elseif exd < 0 && eyd < 0
        ωd = atan(eyd/exd) + pi
    elseif exd > 0 && eyd < 0
        ωd = atan(eyd/exd) + 2*pi
    else
        ωd = atan(eyd/exd)
    end
    Md = ud - ωd
    return [ad;ed;id;Ωd;ωd;Md]
end

function oscOE2meaOE(æ::Vector,dir)
    """
    This function converts computed osculating orbital elements to mean orbit elements.
    However, this algorithm maps only by considering the effect of J2 to the first-order.
    This algorithm works for either oscuOE2meanOE and meanOE2oscuOE depending on 'dir' = True or False
    References:
    [1] Schaub, H., and Junkins, J. L. “Analytical Mechanics Of Space Systems.” 2003.
    https://doi.org/10.2514/4.861550.
    (Appendix F First-Order Mapping Between Mean and Osculating Orbit Elements)
    """
    J2=1.08262668*10^-3
    r_eq=6378137

    a = copy(æ[1])
    e = copy(æ[2])
    i = copy(æ[3])
    Ω = copy(æ[4])
    ω = copy(æ[5])
    M = copy(æ[6])

    gam = 0
    # important parameters
    if dir == true
        gam = -J2/2*(r_eq/a)^2
    else
        gam = J2/2*(r_eq/a)^2
    end

    eta = sqrt(1 - e^2)
    gam_ = gam/eta^4

    E = Kepler_eqn(M+e/2.0,M,e) # eccentric anomaly
    f = 2.0*atan(sqrt((1 + e)/(1 - e))*tan(E/2.0)) # true anomaly
    if f < 0
        f = f + 2*pi
    end

    a_r = (1 + e*cos(f))/eta^2 # ratio of a to r

    # transformed semi-major axis
    a_ = a + a*gam*((3*cos(i)^2 - 1)*(a_r^3  - 1/eta^3) +
        3*(1 - cos(i)^2)*a_r^3*cos(2*ω + 2*f))

    # transformed eccentricity
    de1 = gam_/8.0*e*eta^2*(1 - 11*cos(i)^2 - 40*cos(i)^4/(1 - 5*cos(i)^2))*cos(2*ω)

    de = de1 + eta^2/2*(gam*((3*cos(i)^2 - 1)/eta^6*(e*eta + e/(1 + eta) + 3*cos(f) + 3*e*cos(f)^2 + e^2*cos(f)^3)+
    3*(1 - cos(i)^2)/eta^6*(e + 3*cos(f) + 3*e*cos(f)^2 + e^2*cos(f)^3)*cos(2*ω + 2*f))-
    gam_*(1 - cos(i)^2)*(3*cos(2*ω + f) + cos(2*ω + 3*f)))

    di = -e*de1/(eta^2*tan(i)) + gam_/2*cos(i)*sqrt(1 - cos(i)^2)*(3*cos(2*ω + 2*f) + 3*e*cos(2*ω + f) + e*cos(2*ω + 3*f))

    M_w_W_ = M + ω + Ω + (gam_/8*eta^3*(1 - 11*cos(i)^2 - 40*cos(i)^4/(1 - 5*cos(i)^2)) -
        gam_/16.0*(2 + e^2 - 11*(2 + 3*e^2)*cos(i)^2 -
        40*(2 + 5*e^2)*cos(i)^4/(1 - 5*cos(i)^2) -
        400*e^2*cos(i)^6/(1 - 5*cos(i)^2)^2) +
        gam_/4*(-6*(1 - 5*cos(i)^2)*(f - M + e*sin(f)) +
        (3 - 5*cos(i)^2)*(3 *sin(2*ω + 2*f) + 3*e*sin(2*ω + f) + e*sin(2*ω + 3*f))) -
        gam_/8*e^2*cos(i)*(11 + 80*cos(i)^2/(1 - 5*cos(i)^2) +
        200*cos(i)^4/(1 - 5* cos(i)^2)^2) -
            gam_/2*cos(i)*(6*(f - M + e*sin(f)) - 3*sin(2*ω + 2*f)  -
        3*e*sin(2*ω + f) - e*sin(2*ω + 3*f)))

    edM = gam_/8*e*eta^3*(1 - 11*cos(i)^2 - 40*cos(i)^4/(1 - 5*cos(i)^2))  -
        gam_*eta^3/4*(2*(3*cos(i)^2 - 1)*(a_r^2*eta^2 + a_r + 1)*sin(f) +
        3*(1 - cos(i)^2)*((-a_r^2*eta^2 - a_r + 1)*sin(2*ω + f)+
            (a_r^2*eta^2 + a_r + 1/3)*sin(2*ω + 3*f)))

    dW = -gam_/8.0*e^2*cos(i)*(11 + 80*cos(i)^2/(1 - 5*cos(i)^2) + 200*cos(i)^4/(1 -
    5*cos(i)^2)^2) - gam_/2*cos(i)*(6*(f -
    M + e*sin(f)) - 3*sin(2*ω + 2*f) - 3*e*sin(2*ω + f) - e*sin(2*ω + 3*f))

    d1 = (e + de)*sin(M) + edM*cos(M)
    d2 = (e + de)*cos(M) - edM*sin(M)
    d3 = (sin(i/2) + cos(i/2)*(di/2))*sin(Ω) + sin(i/2)*dW*cos(Ω)
    d4 = (sin(i/2) + cos(i/2)*(di/2))*cos(Ω) - sin(i/2)*dW*sin(Ω)

    M_ = 0
    # transformed mean anomaly
    if d2 < 0 && d1 > 0
        M_ = atan(d1/d2) + pi
    elseif d2 < 0 && d1 < 0
        M_ = atan(d1/d2) + pi
    elseif d2 > 0 && d1 < 0
        M_ = (atan(d1/d2) + 2.0*pi)
    else
        M_ = atan(d1/d2)
    end

    e_ = sqrt(d1^2 + d2^2) # transformed eccentricity

    W_ = 0
    # transformed mean anomaly
    if d4 < 0 && d3 > 0
        W_ = atan(d3/d4) + pi
    elseif d4 < 0 && d3 < 0
        W_ = atan(d3/d4) + pi
    elseif d4 > 0 && d3 < 0
        W_ = (atan(d3/d4) + 2.0*pi)
    else
        W_ = atan(d3/d4)
    end

    i_ = 2*asin(sqrt(d3^2 + d4^2)) # inclination
    w_ = M_w_W_ - M_ - W_ # argument of perigee
    if w_ < 0
        w_ = w_ + 2.0*pi
    end
    return [a_; e_; i_; W_; w_;M_]
end



function mapping_ROE2RTN(æ, δα)
    """
    compute RTN position and velocity vector from orbital elements
    """
    J2 = 1.08262668*10^-3
    μ = 3.986004415*10^14

    a = æ[1]
    ω = æ[5]
    M = æ[6]
    u = ω + M

    n = sqrt(μ/a^3)

    # A Matrix
    A = [1.0  0.0 -cos(u)  -sin(u)  0.0  0.0;
        0  1.0  2.0*sin(u)  -2.0*cos(u)  0  0;
        0  0  0  0  sin(u)  -cos(u);
        0  0  n*sin(u)  -n*cos(u)  0  0;
        -3.0/2.0*n   0  2.0*n*cos(u) 2.0*n*sin(u)  0  0;
        0  0  0  0  n*cos(u)  n*sin(u)]
    return a*A*δα
end
