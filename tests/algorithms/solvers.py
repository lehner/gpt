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
slv_fgcr = w.propagator(inv_pc(eo2, inv.fgcr({"eps": 1e-6, "maxiter": 1000, "restartlen": 20})))
slv_fgmres = w.propagator(inv_pc(eo2, inv.fgmres({"eps": 1e-6, "maxiter": 1000, "restartlen": 20})))
slv_cagcr = w.propagator(inv_pc(eo2, inv.cagcr({"eps": 1e-6, "maxiter": 1000, "restartlen": 10})))
slv_fom = w.propagator(inv_pc(eo2, inv.fom({"eps": 1e-6, "maxiter": 1000, "restartlen": 20})))

# defect-correcting solver at the full field level
slv_dci = w.propagator(
    inv.defect_correcting(inv_pc(eo2, inv.cg({"eps": 1e-40, "maxiter": 25})), eps=1e-6, maxiter=10),
)

# defect-correcting solver at the even-odd level
slv_dci_eo = w.propagator(
    inv_pc(
        eo2,
        inv.defect_correcting(inv.cg({"eps": 1e-40, "maxiter": 25}), eps=1e-6, maxiter=10),
    )
)

# mixed-precision defect-correcting solver at the full field level
slv_dci_mp = w.propagator(
    inv.defect_correcting(
        inv.mixed_precision(inv_pc(eo2, inv.cg({"eps": 1e-40, "maxiter": 25})), g.single, g.double),
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
    resid[name] = eps2**0.5
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
g.message("--------------------------------------------------------------------------------")
g.message("                            Summary of solver tests")
g.message("--------------------------------------------------------------------------------")
g.message("%-38s %-25s %-25s" % ("Solver name", "Solve time / s", "Difference with CG result"))
for t in timings:
    g.message("%-38s %-25s %-25s" % (t, timings[t], resid[t]))


####
# Minimal Residual Extrapolation (Chronological Inverter)
####
g.message("Minimal Residual Extrapolation")
U_eps = [[g(u * rng.element(g.mcolor(U[0].grid), scale=1e-3)) for u in U] for i in range(3)]
w_eps = [w.updated(u) for u in U_eps]

inv_cg = inv.cg({"eps": 1e-8, "maxiter": 500})

solution_space = []
inv_chron = inv.solution_history(
    solution_space,
    inv.sequence(inv.subspace_minimal_residual(solution_space), inv_pc(eo2, inv_cg)),
    2,
)

history = []
for we in w_eps:
    dst_F_we = g(inv_chron(we) * src_F)
    history.append(len(inv_cg.history))

g.message(f"MRE history: {history}")
assert all([h * 1.5 < history[0] for h in history[1:]])


####
# Multi-shift inverters:
####
# the following code also tests that
# the vector -> matrix distribution is
# consistent with multi_shift inverter
# ordering of dst fields.

cg = inv.cg({"eps": 1e-8, "maxiter": 500})
shifts = [0.5, 1.0, 1.7]
mat = eo2_odd(w).Mpc

# also test with multiple sources
src = [rng.cnormal(g.mspincolor(w.F_grid_eo)), rng.cnormal(g.mspincolor(w.F_grid_eo))]
dst_all = g(inv.multi_shift(cg, shifts)(mat).grouped(6) * src)
for i, s in enumerate(shifts):
    for jsrc in range(2):
        eps2 = g.norm2(
            mat * dst_all[2 * i + jsrc] + s * dst_all[2 * i + jsrc] - src[jsrc]
        ) / g.norm2(src[jsrc])
        g.message(f"Test general multi-shift inverter solution: {eps2}")
        assert eps2 < 1e-14

g.default.set_verbose("multi_shift_cg")
mscg = inv.multi_shift_cg({"eps": 1e-8, "maxiter": 1024, "shifts": shifts})

g.default.set_verbose("multi_shift_fom")
msfom = inv.multi_shift_fom({"eps": 1e-8, "maxiter": 1024, "restartlen": 10, "shifts": shifts})

g.default.set_verbose("multi_shift_fgmres")
msfgmres = inv.multi_shift_fgmres(
    {"eps": 1e-8, "maxiter": 1024, "restartlen": 10, "shifts": shifts}
)

prec_fom = inv.multi_shift_fom({"maxiter": 4, "restartlen": 2})
msfgmres_fom = inv.multi_shift_fgmres(
    {"eps": 1e-8, "maxiter": 512, "restartlen": 5, "shifts": shifts, "prec": prec_fom}
)

prec_fgmres = inv.multi_shift_fgmres({"maxiter": 4, "restartlen": 2})
msfgmres_fgmres = inv.multi_shift_fgmres(
    {
        "eps": 1e-8,
        "maxiter": 512,
        "restartlen": 5,
        "shifts": shifts,
        "prec": prec_fgmres,
    }
)


def multi_shift_test(ms, name):
    dst_ms = g(ms(mat) * src)
    for i, s in enumerate(shifts):
        g.message(f"General multi-shift vs multi_shift_{name} for shift {i} = {s}")
        for jsrc in range(2):
            eps2 = g.norm2(dst_all[2 * i + jsrc] - dst_ms[2 * i + jsrc]) / g.norm2(
                dst_ms[2 * i + jsrc]
            )
            g.message(f"Test general solution versus ms{name} solution: {eps2}")
            assert eps2 < 1e-14
            eps2 = g.norm2(
                mat * dst_ms[2 * i + jsrc] + s * dst_ms[2 * i + jsrc] - src[jsrc]
            ) / g.norm2(src[jsrc])
            g.message(f"Test ms{name} inverter solution: {eps2}")
            assert eps2 < 1e-14


multi_shift_test(mscg, "cg")
multi_shift_test(msfom, "fom")
multi_shift_test(msfgmres, "fgmres")
multi_shift_test(msfgmres_fom, "fgmres(fom)")
multi_shift_test(msfgmres_fgmres, "fgmres(fgmres)")
