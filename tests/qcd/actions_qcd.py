#!/usr/bin/env python3
#
# Authors: Mattia Bruno 2020
#          Christoph Lehner 2020
#
# Desc.: Illustrate core concepts and features
#

import gpt as g

g.default.set_verbose("random", False)

# gauge field
rng = g.random("test")
grid = g.grid([8] * 4, g.double)

# wilson
p={
    #"kappa" : 0.137,
    "mass" : 0.1,
    "csw_r" : 0,
    "csw_t" : 0,
    "xi_0" : 1,
    "nu" : 1,
    "isAnisotropic" : False,
    "boundary_phases" : [ 1,1,1,-1 ]
}
wilson = g.qcd.fermion.wilson_clover

act = g.qcd.actions.fermion.doublet

eo2_ne = g.qcd.fermion.preconditioner.eo2_ne
inv = g.algorithms.inverter
inv1 = inv.preconditioned(eo2_ne, inv.cg({"eps": 1e-8, "maxiter": 1024}))

# pseudo-fermion field
pf = g.vspincolor(grid)

# first test: free field
U = g.qcd.gauge.unit(grid)
Dop = wilson(U, p)
a = act(pf, Dop, inv1, inv1)

a.refresh(rng)
assert a() < 1e-12
g.message(f"Free Action {a()} , type {acts[i]}")

import sys
sys.exit()

# second test: gauge invariance
V = rng.lie(g.mcolor(grid))
Up = g.qcd.gauge.unit(grid)
Um = g.qcd.gauge.unit(grid)

for i in range(len(acts)):
    a0 = acts[i](U, beta[i])
    a0.hot_start(rng)
    da = a0()

    a1 = acts[i](g.qcd.gauge.transformed(U, V), beta[i])
    da -= a1()

    assert abs(da) < 1e-10
    g.message(f"Difference after gauge trafo {da:g} , type {acts[i]}")

# third test: force numerical vs analytical
eps = 1.0e-4

mom = g.algorithms.markov.conjugate_momenta(U)
mom.refresh(rng)

frc = g.lattice(U[0])

for i in range(len(acts)):
    for mu in range(grid.nd):
        Up[mu] @= g.core.matrix.exp(g.eval(+eps * mom.mom[mu])) * U[mu]
        Um[mu] @= g.core.matrix.exp(g.eval(-eps * mom.mom[mu])) * U[mu]

    ap = acts[i](Up, beta[i])
    am = acts[i](Um, beta[i])
    da = 2 / 3 * (ap() - am())

    for mu in range(grid.nd):
        Up[mu] @= g.core.matrix.exp(g.eval(+2.0 * eps * mom.mom[mu])) * U[mu]
        Um[mu] @= g.core.matrix.exp(g.eval(-2.0 * eps * mom.mom[mu])) * U[mu]

    ap = acts[i](Up, beta[i])
    am = acts[i](Um, beta[i])
    da += 1 / 12 * (-ap() + am())

    da *= 1 / eps

    aref = acts[i](U, beta[i])
    aref.setup_force()
    daa = 0.0
    for mu in range(grid.nd):
        frc @= aref.force(U[mu])
        daa += (
            2.0 * g.inner_product(frc, mom.mom[mu]).real
        )  # factor 2 due to normalization of algebra
    assert abs(da / daa - 1.0) < eps ** 2
    g.message(f"Force difference {abs(da/daa-1.0)} , type {acts[i]}")
