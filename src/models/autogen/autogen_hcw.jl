"""
    HCW dynamics auto generation script
"""

using LinearAlgebra
using Symbolics

# define variables
@variables t ω μ x y z vx vy vz


f1 = vx
f2 = vy
f3 = vz
f4 = 3 * ω^2 * x + 2 * ω * vy
f5 = -2 * ω * vx
f6 = -ω^2 * z

f = [f1, f2, f3, f4, f5, f6]
df1 = Symbolics.derivative(f, [x, y, z, vx, vy, vz])
println(df1)

df2 = Symbolics.jacobian(f, [x, y, z, vx, vy, vz])
println(df2)