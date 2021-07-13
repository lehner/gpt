#!/usr/bin/env python3
#
# Authors: Christoph Lehner, Mattia Bruno 2020
#
# Desc.: Illustrate core concepts and features
#
import gpt as g
import numpy as np

grid = g.grid([8,8,8,8],g.single)
rng = g.random("deriv")

U = g.qcd.gauge.unit(grid)
rng.normal_element(U)

# wilson parameters
# p = {
#     "kappa": 0.137,
#     "csw_r": 0.0,
#     "csw_t": 0.0,
#     "xi_0": 1,
#     "nu": 1,
#     "isAnisotropic": False,
#     "boundary_phases": [1., 1., 1., 1.],
# }
# M = g.qcd.fermion.wilson_clover(U, p)
mobius_params = {
    "mass": 0.08,
    "M5": 1.8,
    "b": 1.5,
    "c": 0.5,
    "Ls": 12,
    "boundary_phases": [1.0, 1.0, 1.0, 1.0],
}
M = g.qcd.fermion.mobius(U, mobius_params)

psi = []
for _ in [0,1]:
    psi.append(g.vspincolor(M.F_grid))
    rng.normal(psi[-1])

psi[0] @= psi[1]

mom = g.group.cartesian(U)
rng.normal_element(mom)

frc0 = g.group.cartesian(U)
M.gradient(frc0, g(M * psi[0]), psi[1])
M.gradientDag(mom, psi[0], g(M * psi[1]))
for mu in range(grid.nd):
   frc0[mu] @= frc0[mu] + mom[mu]

rng.normal_element(mom)

dS_ex = 0.0
for mu in range(grid.nd):
    #dS_ex += g.sum(g.trace(frc0[mu] * mom[mu])) * 2.0
    dS_ex += g.group.inner_product(frc0[mu], mom[mu])

#### numerical derivative
Uprime = g.qcd.gauge.unit(grid)
eps = 1e-2

dS_num = 0.0
for coeff in [(1.0, 2./3.), (2.0, -1./12.), (-1.0,-2./3.), (-2.0,1/12)]:
#for coeff in [(1.0, 0.5), (-1.0, -0.5)]:
    f = eps * coeff[0]
    for mu in range(grid.nd):
        Uprime[mu] @= g.group.compose(g(f * mom[mu]), U[mu])

    Mprime = M.updated(Uprime)
    dS_num += (coeff[1]/eps) * g.inner_product(g(Mprime * psi[0]), g(Mprime * psi[1]))

g.message(f"Exact gradient {dS_ex:.6e}")
g.message(f"Numer gradient {dS_num:.6e}")
g.message(f"Difference {abs(dS_num - dS_ex)/abs(dS_ex):.6e}")

assert abs(dS_num - dS_ex)/abs(dS_ex) < 1e-4