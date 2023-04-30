# CDDP

[![Build Status](https://github.com/astomodynamics/CDDP.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/astomodynamics/CDDP.jl/actions/workflows/CI.yml?query=branch%3Amain)

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