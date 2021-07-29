#!/usr/bin/env python3
#
# Authors: Christoph Lehner, Mattia Bruno 2020
#
# Desc.: Illustrate core concepts and features
#
import gpt as g
import numpy as np

g.default.set_verbose("cg", False)

grid = g.grid([8, 8, 8, 8], g.double)
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
    "boundary_phases": [1.0, 1.0, 1.0, 1.0],
}
M = g.qcd.fermion.wilson_clover(U, p)

p = {
    "kappa": 0.13465,
    "csw_r": 0.0,
    "csw_t": 0.0,
    "xi_0": 1,
    "nu": 1,
    "isAnisotropic": False,
    "boundary_phases": [1.0, 1.0, 1.0, 1.0],
}
M2 = g.qcd.fermion.wilson_clover(U, p)


psi = g.vspincolor(M.F_grid)
psi_o = g.vspincolor(M.F_grid_eo)
rng.normal(psi)
g.pick_checkerboard(g.odd, psi_o, psi)

inv = g.algorithms.inverter
sol = inv.cg({"eps": 1e-10, "maxiter": 1024})
a = g.qcd.pseudofermion.action

acts = []
acts += [(a.two_flavor(M, sol), "two_flavor", psi)]
acts += [(a.two_flavor_evenodd(M, sol), "two_flavor_evenodd", psi)]
acts += [(a.two_flavor_ratio([M, M2], sol), "two_flavor_ratio", psi)]
acts += [
    (
        a.two_flavor_ratio_evenodd_schur([M, M2], sol),
        "two_flavor_ratio_evenodd_schur",
        psi_o,
    )
]

for _a in acts:
    a, name, pf = _a
    g.message(name)
    fields = U + [pf]
    da = a.draw(fields, rng) - a(fields)
    g.message(f"difference action drawn vs computed: da = {da:g}")
    assert abs(da) < 1e-7
    a.assert_gradient_error(rng, fields, U, 1e-3, 5e-7)
