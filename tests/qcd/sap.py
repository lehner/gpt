#!/usr/bin/env python3
#
# Authors: Mattia Bruno 2020
#          Christoph Lehner 2020
#
# Desc.: Illustrate core concepts and features
#
import gpt as g
import numpy as np
import sys


# gauge field
rng = g.random("test")
U = g.qcd.gauge.random(g.grid([16, 16, 16, 32], g.single), rng)

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
inv = g.algorithms.inverter
inv_pc = inv.preconditioned
pc = g.qcd.fermion.preconditioner

# solver used to solve dirac equation on SAP blocks
mr = inv.mr({"eps": 1e-16, "maxiter": 4, "relax": 1})
g.default.set_verbose("mr", False)

# sap inverter
inv1 = inv.defect_correcting(
    pc.sap_cycle(inv_pc(pc.eo2(), mr), bs=[4, 4, 4, 4]), eps=1e-6, maxiter=20,
)

# point source
src = g.vspincolor(w.F_grid)
src[:] = 0
src[0, 0, 0, 0] = g.vspincolor([[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]])
dst = g.lattice(src)
inv1_w = inv1(w)
t0 = g.time()
dst = g.eval(inv1_w * src)
t1 = g.time()

# use different solver and compare
fgcr = inv.fgcr({"eps": 1e-7, "maxiter": 1024, "restartlen": 8})
inv2 = inv_pc(pc.eo2(), fgcr)

dst2 = g.lattice(src)
inv2_w = inv2(w)
t2 = g.time()
dst2 = g.eval(inv2_w * src)
t3 = g.time()

rr = g.norm2(dst2 - dst)
g.message(
    f"Difference of results: {rr}, Time for SAP-based-solve: {t1-t0} s, Time for FGCR: {t3-t2} s"
)
assert rr < 1e-11
