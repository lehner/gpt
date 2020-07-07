#!/usr/bin/env python3
#
# Authors: Daniel Richtmann 2020
#          Christoph Lehner 2020
#
# Desc.: Test multigrid for clover
#
import gpt as g
import numpy as np
import sys

# load configuration
precision = g.double
U = g.qcd.gauge.random(g.grid([8, 8, 8, 8], precision), g.random("test_alg"))

# use the gauge configuration grid
grid_f = U[0].grid

# create the coarse grid
grid_c = g.grid([4, 4, 4, 4], grid_f.precision)

# quark
w_dp = g.qcd.fermion.wilson_clover(
    U,
    {
        "mass": -0.25,
        "csw_r": 2.0171,
        "csw_t": 2.0171,
        "xi_0": 1,
        "nu": 1,
        "isAnisotropic": False,
        "boundary_phases": [1.0, 1.0, 1.0, 1.0],
    },
)

# create point source
src = g.vspincolor(grid_f)
src[0, 1, 0, 0] = g.vspincolor([[1] * 3] * 4)

# abbreviations
s = g.qcd.fermion.solver
a = g.algorithms.iterative
p = g.qcd.fermion.preconditioner
w_sp = w_dp.converted(g.single)

# preconditioners, for params can give
# lists (separate values for each level, length must be compatible with number of grids (asserted inside))
# scalar value (broadcast to all levels)
mg = a.mg(
    w_dp,
    {
        # "grid": [grid_f, grid_c, grid_c, grid_c],
        "grid": [grid_f, grid_c],
        # alternatively, we should be able to give one grid + a list of block sizes
        "northo": 2,
        "nbasis": 10,
        "hermitian": True,
        "vecstype": "test",
        "presmooth": a.mr({"eps": 1e-1, "maxiter": 16, "relax": 1}),
        # "presmooth": None,
        "postsmooth": a.mr({"eps": 1e-1, "maxiter": 16, "relax": 1}),
        "coarsestsolve": a.mr({"eps": 1e-1, "maxiter": 16, "relax": 1}),
        "setupsolve": a.mr({"eps": 1e-1, "maxiter": 16, "relax": 1}),
    },
)
eo2_even_dp = p.eo2(w_dp, parity=g.even)

# solver params
eps = 1e-8
maxiter = 1000
restartlen = 20

# solver algorithms
algs = {
    "cg": a.cg({"eps": eps, "maxiter": maxiter}),
    "fgcr_none": a.fgcr(
        {"eps": eps, "maxiter": maxiter, "restartlen": restartlen, "prec": None}
    ),
    "fgcr_mg": a.fgcr(
        {"eps": eps, "maxiter": maxiter, "restartlen": restartlen, "prec": mg()}
    ),
    "fgmres_none": a.fgmres(
        {"eps": eps, "maxiter": maxiter, "restartlen": restartlen, "prec": None}
    ),
    "fgmres_mg": a.fgmres(
        {"eps": eps, "maxiter": maxiter, "restartlen": restartlen, "prec": mg()}
    ),
}

# cg as reference
slv_cg_eo = s.propagator(s.inv_eo_ne(eo2_even_dp, algs["cg"]))
dst_cg = g.eval(slv_cg_eo * src)

# helper dictionaries
timings = {}
resid = {}
iters = {}
resid_cg = {}


def test_alg(alg, name):
    if name == "cg":
        return
    print("Starting with solver %s" % name)
    t0 = g.time()
    dst = g.eval(alg(w_dp.M) * src)
    t1 = g.time()
    timings[name] = t1 - t0
    resid[name] = (g.norm2(w_dp.M * dst - src) / g.norm2(src)) ** 0.5
    resid_cg[name] = (g.norm2(dst_cg - dst) / g.norm2(dst_cg)) ** 0.5
    iters[name] = len(alg.history)


# run solvers
for k, v in algs.items():
    test_alg(v, k)

# print statistics
g.message(
    "--------------------------------------------------------------------------------"
)
g.message("                            Summary of solver tests")
g.message(
    "--------------------------------------------------------------------------------"
)
g.message(
    "%-38s %-25s %-25s %-25s %-25s"
    % (
        "Solver name",
        "Solve time / s",
        "Solve iterations",
        "Residual",
        "Difference with CG result",
    )
)
for k, v in timings.items():
    g.message("%-38s %-25g %-25d %-25g %-25g" % (k, v, iters[k], resid[k], resid_cg[k]))

# print mg profiling
g.message(
    "Contributions to time spent in MG preconditioner (possibly accumulated for all mg solver instances with same preconditioner)"
)
for lvl in reversed(range(len(mg.t_setup))):
    mg.t_setup[lvl].print("mg_setup_lvl_" + str(lvl))
for lvl in reversed(range(len(mg.t_solve))):
    mg.t_solve[lvl].print("mg_solve_lvl_" + str(lvl))
