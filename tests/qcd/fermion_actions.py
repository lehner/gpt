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


def mobius(m_plus, m_minus):
    return g.qcd.fermion.mobius(
        U,
        mass_plus=m_plus,
        mass_minus=m_minus,
        M5=1.8,
        b=1.5,
        c=0.5,
        Ls=6,
        boundary_phases=[1, 1, 1, -1],
    )


psi = g.vspincolor(M.F_grid)
psi_o = g.vspincolor(M.F_grid_eo)
rng.normal(psi)
g.pick_checkerboard(g.odd, psi_o, psi)

inv = g.algorithms.inverter
pc = g.qcd.fermion.preconditioner
sol = inv.cg({"eps": 1e-10, "maxiter": 1024})
sol_pc = inv.preconditioned(pc.eo2_ne(), sol)
a = g.qcd.pseudofermion.action

rat = g.algorithms.rational.zolotarev_inverse_square_root(1.0**0.5, 3**0.5, 7)
rat_fnc = g.algorithms.rational.rational_function(rat.zeros, rat.poles, rat.norm)

acts = []
acts += [
    (
        a.exact_one_flavor_ratio(mobius, 0.5, 1.0, sol_pc),
        "exact_one_flavor_ratio",
        psi,
        [rat_fnc],
    )
]
acts += [(a.two_flavor(M, sol), "two_flavor", psi, [])]
acts += [(a.two_flavor_evenodd(M, sol), "two_flavor_evenodd", psi, [])]
acts += [(a.two_flavor_ratio([M, M2], sol), "two_flavor_ratio", psi, [])]
acts += [
    (
        a.two_flavor_ratio_evenodd_schur([M, M2], sol),
        "two_flavor_ratio_evenodd_schur",
        psi_o,
        [],
    )
]

sm = g.qcd.gauge.smear.stout(rho=0.157)
a_sm = acts[0][0].transformed(sm)
a_sm.assert_gradient_error(rng, U + [acts[0][2]], U, 1e-3, 5e-7)

for _a in acts:
    a, name, pf, dargs = _a
    g.message(name)
    fields = U + [pf]
    a_fields_draw = a.draw(fields, rng, *dargs)
    a_fields = a(fields)
    da = a_fields_draw - a_fields
    g.message(f"difference action drawn vs computed: da = {da:g}")
    assert abs(da) < 1e-7
    a.assert_gradient_error(rng, fields, U, 1e-3, 5e-7)

    # test action double precision versus quadruple precision
    fields_quad = g.convert(fields, g.double_quadruple)
    a_fields_quad = a(fields_quad)
    eps = abs((float(a_fields_quad) - a_fields) / a_fields)
    g.message(f"quad precision regression against double precision: {eps}")
    assert eps < 1e-14
