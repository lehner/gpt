#!/usr/bin/env python3
#
# Authors: Mattia Bruno 2020
#
# Desc.: Illustrate core concepts and features
#
import gpt as g
import numpy as np
import sys


# gauge field
rng = g.random("test")
U = g.qcd.gauge.random(g.grid([8, 8, 8, 16], g.single), rng)

# wilson
p = {
    # "kappa" : 0.137,
    "mass": 0.1,
    "csw_r": 0,
    "csw_t": 0,
    "xi_0": 1,
    "nu": 1,
    "isAnisotropic": False,
    "boundary_phases": [1, 1, 1, -1],
}
w = g.qcd.fermion.wilson_clover(U, p)

# shortcuts
s = g.qcd.fermion.solver
pc = g.qcd.fermion.preconditioner

# sap preconditioner
sap = pc.sap(w, bs=[4, 4, 4, 4])

# solver used to solve dirac equation on SAP blocks
mr = g.algorithms.iterative.mr({"eps": 1e-16, "maxiter": 4, "relax": 1})
g.default.set_verbose("mr", False)


def blk_solver(x):
    return s.inv_eo(pc.eo2(x), mr)


# sap inverter
inv = s.inv_sap(sap, blk_solver, ncy=20)

# point source
src = g.vspincolor(w.F_grid)
src[0, 0, 0, 0] = g.vspincolor([[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]])
dst = g.lattice(src)
dst = g.eval(inv * src)

# use different solver and compare
fgcr = g.algorithms.iterative.fgcr({"eps": 1e-7, "maxiter": 1024, "restartlen": 8})
inv2 = s.inv_eo(pc.eo2(w), fgcr)

dst2 = g.lattice(src)
dst2 = g.eval(inv2 * src)

rr = g.norm2(dst2 - dst)
g.message(f"{rr}")
assert rr < 1e-13
