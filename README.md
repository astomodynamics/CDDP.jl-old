# CDDP

Julia implementation of the constrained differential dynamic programming algorithm for solving the optimal control problem of nonlinear systems with state and control constraints.

## Installation
```zsh
git clone https://github.com/astomodynamics/CDDP.jl
cd CDDP.jl
cd src
julia> ]
  (@v1.9) pkg> activate ..
  (CDDP) pkg> dev "../../CDDP.jl"
  (CDDP) pkg> precompile
```

## Examples
1. Linear spacecraft relative motion control

## References
1. Pavlov, A., Shames, I., and Manzie, C. “Interior Point Differential Dynamic Programming.” IEEE Transactions on Control Systems Technology, Vol. 29, No. 6, 2021, pp. 2720–2727. https://doi.org/10.1109/tcst.2021.3049416.
2. T. Sasaki, K. Ho, E. G. Lightsey, "Nonlinear Spacecraft Formation Flying using Constrained Differential Dynamic Programming," in Proceedings of AAS/AIAA Astrodynamics Specialist Conference, 2022. https://ssdl.gatech.edu/sites/default/files/ssdl-files/papers/conferencePapers/AAS-22-795.pdf.
