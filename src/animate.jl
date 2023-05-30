
using Plots; gr()

function make_gif(ode_sol, dt, model; example=:cartpole, filename="../results/example.gif", fps=30)
    anime = Animation()
    if example==:cartpole
        params = model.params
        mc = params.mc
        mp = params.mp
        l = params.l
        g = params.g
        tf = ode_sol.t[end]
        for k in 1:Int64(tf/dt)
            x = ode_sol(k*dt)
            xc = x[1]
            θ = x[2]
            xx = l*sin(θ)
            xy = -l*cos(θ)
            cart_height = 0.15
            r_cart = [xc, cart_height ]
            plt = plot([r_cart[1], xx+r_cart[1]], [r_cart[2], xy+r_cart[2]], label="",xlim=(-1.5, 1.5), ylim=(-1.5, 1.5), color=:blue, linewidth=2, equal_aspect=true)
            
            scatter!((r_cart[1],r_cart[2]), color=:red, markersize=6, legend=false, marker=:circle, fillcolor=nothing)
            # mass point
            scatter!((xx+r_cart[1], xy+r_cart[2]), color=:black, markersize=6, legend=false, marker=:square, fillcolor=nothing)
            # horizontal line
            plot!([-1.5, 1.5], [0., 0.], label="", color=:black, linewidth=1)
            
            # right wall
            plot!([r_cart[1]+0.2, r_cart[1]+0.2], [0., 0.3], label="", color=:black, linewidth=1)
            # left wall
            plot!([r_cart[1]-0.2, r_cart[1]-0.2], [0., 0.3], label="", color=:black, linewidth=1)
            # top wall
            plot!([r_cart[1]-0.2, r_cart[1]+0.2], [0.3, 0.3], label="", color=:black, linewidth=1)
            # bottom wall
            plot!([r_cart[1]-0.2, r_cart[1]+0.2], [0., 0.], label="", color=:black, linewidth=1)
            plot!(aspect_ratio=:equal)
            plot!(xlim=(-1.0, 1.0), ylim=(-1.0, 1.0))
            plot!(title="Cartpole")
            frame(anime, plt)
        end
        

    elseif example==:pendulum
        nothing
    end    

    gif(anime, filename, fps = fps)
end
