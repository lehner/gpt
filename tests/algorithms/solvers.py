#!/usr/bin/env python3
#
# Authors: Daniel Richtmann 2020
#          Christoph Lehner 2020
#
# Desc.: Exercise linear solvers
#
import gpt as g
import numpy as np
import sys
import time
import os.path

# load configuration
precision = g.double
rng = g.random("test")
U = g.qcd.gauge.random(g.grid([8, 8, 8, 16], precision), rng)

# use the gauge configuration grid
grid = U[0].grid

# quark
w = g.qcd.fermion.wilson_clover(
    U,
    {
        "kappa": 0.13565,
        "csw_r": 2.0171 / 2.0,  # for now test with very heavy quark
        "csw_t": 2.0171 / 2.0,
        "xi_0": 1,
        "nu": 1,
        "isAnisotropic": False,
        "boundary_phases": [1.0, 1.0, 1.0, 1.0],
    },
)

# create point source
src = g.vspincolor(grid)
src[:] = 0
src[0, 1, 0, 0] = g.vspincolor([[1] * 3] * 4)

# build solvers
inv = g.algorithms.inverter
inv_pc = inv.preconditioned
ac = g.algorithms.assert_converged

eo2_odd = g.qcd.fermion.preconditioner.eo2_ne(parity=g.odd)
eo2_even = g.qcd.fermion.preconditioner.eo2_ne(parity=g.even)
eo1_odd = g.qcd.fermion.preconditioner.eo1_ne(parity=g.odd)
eo1_even = g.qcd.fermion.preconditioner.eo1_ne(parity=g.even)
# default
eo2 = eo2_odd
eo2_sp = eo2_odd

# test full-site guess for preconditioner
src_F = g.vspincolor(w.F_grid)
src_F[:] = 0
src_F[0, 1, 0, 0] = g.vspincolor([[1] * 3] * 4)
eo2_inv = inv_pc(eo2, ac(inv.cg({"eps": 1e-8, "maxiter": 500})))(w)
dst_F = g(eo2_inv * src_F)
for pc in [eo1_odd, eo1_even, eo2_odd, eo2_even]:
    cg = inv.cg({"eps": 1e-7, "maxiter": 1000})
    gen_inv = inv_pc(pc, cg)(w)
    dst_gen = g.copy(dst_F)
    gen_inv(dst_gen, src_F)
    # make sure using the guess results in immediate convergence
    assert len(cg.history) == 1

# run with higher stopping condition since it will be the reference run
slv_cg = w.propagator(inv_pc(eo2, inv.cg({"eps": 1e-8, "maxiter": 1000})))
# other pc and parity
slv_cg_eo2_even = w.propagator(inv_pc(eo2_even, inv.cg({"eps": 1e-8, "maxiter": 1000})))
slv_cg_eo1_odd = w.propagator(inv_pc(eo1_odd, inv.cg({"eps": 1e-8, "maxiter": 1000})))
slv_cg_eo1_even = w.propagator(inv_pc(eo1_even, inv.cg({"eps": 1e-8, "maxiter": 1000})))
# other parity/pc
slv_cg = w.propagator(inv_pc(eo2, inv.cg({"eps": 1e-8, "maxiter": 1000})))

# solvers to test against CG
slv_mr = w.propagator(inv_pc(eo2, inv.mr({"eps": 1e-6, "maxiter": 1000, "relax": 1.0})))
slv_bicgstab = w.propagator(inv_pc(eo2, inv.bicgstab({"eps": 1e-6, "maxiter": 1000})))
slv_fgcr = w.propagator(
    inv_pc(eo2, inv.fgcr({"eps": 1e-6, "maxiter": 1000, "restartlen": 20}))
)
slv_fgmres = w.propagator(
    inv_pc(eo2, inv.fgmres({"eps": 1e-6, "maxiter": 1000, "restartlen": 20}))
)
slv_cagcr = w.propagator(
    inv_pc(eo2, inv.cagcr({"eps": 1e-6, "maxiter": 1000, "restartlen": 10}))
)
slv_fom = w.propagator(
    inv_pc(eo2, inv.fom({"eps": 1e-6, "maxiter": 1000, "restartlen": 20}))
)

# defect-correcting solver at the full field level
slv_dci = w.propagator(
    inv.defect_correcting(
        inv_pc(eo2, inv.cg({"eps": 1e-40, "maxiter": 25})), eps=1e-6, maxiter=10
    ),
)

# defect-correcting solver at the even-odd level
slv_dci_eo = w.propagator(
    inv_pc(
        eo2,
        inv.defect_correcting(
            inv.cg({"eps": 1e-40, "maxiter": 25}), eps=1e-6, maxiter=10
        ),
    )
)

# mixed-precision defect-correcting solver at the full field level
slv_dci_mp = w.propagator(
    inv.defect_correcting(
        inv.mixed_precision(
            inv_pc(eo2, inv.cg({"eps": 1e-40, "maxiter": 25})), g.single, g.double
        ),
        eps=1e-6,
        maxiter=10,
    )
)

# perform solves (reference)
dst_cg = g.eval(slv_cg * src)
g.message("CG finished")

timings = {}
resid = {}


def test(slv, name):
    t0 = g.time()
    dst = g.eval(slv * src)
    t1 = g.time()
    eps2 = g.norm2(dst_cg - dst) / g.norm2(dst_cg)
    g.message("%s finished: eps^2(CG) = %g" % (name, eps2))
    timings[name] = t1 - t0
    resid[name] = eps2 ** 0.5
    assert eps2 < 5e-7


test(slv_cg_eo2_even, "CG eo2_even")
test(slv_cg_eo1_even, "CG eo1_even")
test(slv_cg_eo1_odd, "CG eo1_odd")
test(slv_dci, "Defect-correcting solver")
test(slv_dci_eo, "Defect-correcting (eo)")
test(slv_dci_mp, "Defect-correcting (mixed-precision)")
test(slv_mr, "MR")
test(slv_bicgstab, "BICGSTAB")
test(slv_fgcr, "FGCR")
test(slv_fgmres, "FGMRES")
test(slv_cagcr, "CAGCR")
test(slv_fom, "FOM")

# summary
g.message(
    "--------------------------------------------------------------------------------"
)
g.message("                            Summary of solver tests")
g.message(
    "--------------------------------------------------------------------------------"
)
g.message(
    "%-38s %-25s %-25s" % ("Solver name", "Solve time / s", "Difference with CG result")
)
for t in timings:
    g.message("%-38s %-25s %-25s" % (t, timings[t], resid[t]))


# Multi-shift inverters:
# the following code also tests that
# the vector -> matrix distribution is
# consistent with multi_shift inverter
# ordering of dst fields.
cg = inv.cg({"eps": 1e-8, "maxiter": 500})
shifts = [0.2, 0.5, 1.0]
mat = eo2_odd(w).Mpc
src_eo = rng.cnormal(g.mspincolor(w.F_grid_eo))
dst_eo_all = g(inv.multi_shift(cg, shifts)(mat) * src_eo)
for i, s in enumerate(shifts):
    sinv = cg(lambda dst, src: g.eval(dst, mat * src + s * src))
    dst_eo = g(sinv * src_eo)
    eps2 = g.norm2(dst_eo - dst_eo_all[i]) / g.norm2(dst_eo)
    g.message(
        f"Test general multi-shift inverter against individual inversions: {eps2}"
    )
    assert eps2 < 1e-15

# TODO: add specialized multi-shift inverters and tests
g.default.set_verbose("multi_shift_cg")

# multi-shift cg like cg works with vspincolor
src_eo = rng.cnormal(g.vspincolor(w.F_grid_eo))
dst = g(inv.multi_shift(cg, shifts)(mat) * src_eo)

mscg = inv.multi_shift_cg({"eps": 1e-8, "maxiter": 1024, "shifts": shifts})
dst_mscg = g(mscg(mat) * src_eo)

def mat_shift(dst, src, s):
    dst @= mat * src + s * src

for i, s in enumerate(shifts):
    g.message(f"General multi-shift vs multi_shift_cg for shift {i} = {s}")
    eps2 = g.norm2(dst[i] - dst_mscg[i]) / g.norm2(dst_mscg[i])
    g.message(f"Test solutions: {eps2}")
    tmp = g.lattice(src_eo)
    mat_shift(tmp, dst[i], shifts[i])
    g.message(f"Test 1 - [mat + s][mat + s]^(-1) = {g.norm2(tmp) - g.norm2(src_eo)}")
    
