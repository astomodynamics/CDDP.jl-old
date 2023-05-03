# CDDP

Julia implementation of the constrained differential dynamic programming algorithm for solving the optimal control problem of nonlinear systems with state and control constraints.

## Installation
```zsh
git clone https://github.com/astomodynamics/CDDP.jl
cd CDDP.jl
cd src
julia> ]
  (@v1.8) pkg> activate ..
  (CDDP) pkg> dev ../../CDDP.jl/
  (CDDP) pkg> precompile
```

## Examples
1. Linear spacecraft relative motion control
