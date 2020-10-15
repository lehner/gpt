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

eo2_ne = g.qcd.fermion.preconditioner.eo2_ne(parity=g.odd)
inv = g.algorithms.inverter
inv1 = inv.preconditioned(eo2_ne, inv.cg({"eps": 1e-12, "maxiter": 1024}))
inv2 = inv.preconditioned(eo2_ne, inv.cg({"eps": 1e-10, "maxiter": 1024}))

# pseudo-fermion field
pf = g.vspincolor(grid)

# first test: free field
U = g.qcd.gauge.random(grid, rng)
Dop = wilson(U, p)
a = act(pf, Dop, inv1, inv2)

a0 = a(rng)
a1 = a()
print(a0 - a1)
assert abs(a0 - a1) * 12 < 1e-8

# force numerical vs analytical
eps = 1.0e-4

mom = g.algorithms.markov.conjugate_momenta(U)
mom.refresh(rng)

frc = g.lattice(U[0])

Up = g.qcd.gauge.unit(grid)
Um = g.qcd.gauge.unit(grid)

for mu in range(grid.nd):
    Up[mu] @= g.core.matrix.exp(g.eval(+eps * mom.mom[mu])) * U[mu]
    Um[mu] @= g.core.matrix.exp(g.eval(-eps * mom.mom[mu])) * U[mu]

Dop.update(Up)
ap = a()

Dop.update(Um)
am = a()
da = 2 / 3 * (ap - am)

for mu in range(grid.nd):
    Up[mu] @= g.core.matrix.exp(g.eval(+2.0 * eps * mom.mom[mu])) * U[mu]
    Um[mu] @= g.core.matrix.exp(g.eval(-2.0 * eps * mom.mom[mu])) * U[mu]

Dop.update(Up)
ap = a()

Dop.update(Um)
am = a()
da += 1 / 12 * (-ap + am)

da *= 1 / eps

a.setup_force()
daa = 0.0
for mu in range(grid.nd):
    frc @= a.force(U[mu])
    daa += (
        2.0 * g.inner_product(frc, mom.mom[mu]).real
    )  # factor 2 due to normalization of algebra
print(da,daa)
a.clean_force()
#assert abs(da / daa - 1.0) < eps ** 2
#g.message(f"Force difference {abs(da/daa-1.0)} , type {acts[i]}")
