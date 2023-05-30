
struct ConstraintFunction <: AbstractDDPFunction
    c::Function
    ∇c::Function
    ∇²c::Function
    function ConstFunc(;
        c=empty,
        ∇c=empty,
        ∇²c=empty,
    )
        new(
            c,
            ∇c,
            ∇²c,
        )
    end
end

"""
    get_feasibility(model, X, U)
Check feasibility
"""
function get_feasibility(
    prob::AbstractDDPProblem,
    X,
    U,
)
    dt = prob.dt
    for k in 0:prob.tN-1
        c = prob.c(X(k*dt), U(k*dt))
        for c_ele in c
            if c_ele >= 0
                return false
            end
        end
    end
    return true
end