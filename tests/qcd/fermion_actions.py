#!/usr/bin/env python3
#
# Authors: Christoph Lehner, Mattia Bruno 2020
#
# Desc.: Illustrate core concepts and features
#
import gpt as g
import numpy as np

g.default.set_verbose('cg',False)

grid = g.grid([8,8,8,8], g.double)
rng = g.random("deriv")

U = g.qcd.gauge.unit(grid)
rng.normal_element(U)

p = {
    "kappa": 0.13565,
    "csw_r": 0.0,
    "csw_t": 0.0,
    "xi_0": 1,
    "nu": 1,
    "isAnisotropic": False,
    "boundary_phases": [1., 1., 1., 1.],
}
M = g.qcd.fermion.wilson_clover(U, p)

psi = g.vspincolor(grid)
rng.normal(psi)

inv = g.algorithms.inverter
sol = inv.cg({"eps": 1e-7, "maxiter": 1024})

acts = []
acts += [g.qcd.scalar.action.quadratic_MDagM(M, sol)]

for a in acts:
    da = a.draw(U + [psi], rng) - a(U + [psi])
    g.message(f'difference action drawn vs computed: da = {da:g}')
    assert abs(da) < 1e-9
    a.assert_gradient_error(rng, U+[psi], U, 1e-4, 5e-7)