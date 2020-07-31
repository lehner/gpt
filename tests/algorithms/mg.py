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

# setup rng
rng = g.random("test_mg")

# setup gauge field
U = g.qcd.gauge.random(g.grid([16, 16, 16, 16], g.double), rng)

# use the gauge configuration grid
grid_f = U[0].grid

# quark
w_dp = g.qcd.fermion.wilson_clover(
    U,
    {
        "mass": -0.2,
        "csw_r": 1.0,
        "csw_t": 1.0,
        "xi_0": 1,
        "nu": 1,
        "isAnisotropic": False,
        "boundary_phases": [1.0, 1.0, 1.0, -1.0],
    },
)

# create source
src = g.vspincolor(grid_f)
src[0, 1, 0, 0] = g.vspincolor([[1] * 3] * 4)
# rng.cnormal(src)

# create reference solution
dst_cg_eone_none = g.copy(src)
dst_cg_eone_none[:] = 0

# abbreviations
s = g.qcd.fermion.solver
a = g.algorithms.iterative
p = g.qcd.fermion.preconditioner
w_sp = w_dp.converted(g.single)

# mg params structure:
# if a scalar value (= not a list) is given, the parameter is broadcast to every level
# specifying a list for a parameter instead enables explicit configuration of every level
# the length of the lists must then be compatible with the number of grids (asserted inside)
mg_params_2lvl = {
    "grid": [grid_f, g.block.grid(grid_f, [2, 2, 2, 2])],
    "northo": 1,
    "nbasis": 40,
    "hermitian": False,
    "savelinks": True,
    "vecstype": "test",
    "presmooth": None,
    "postsmooth": lambda mat: s.inv_eo(
        p.eo2(mat), a.bicgstab({"eps": 1e-1, "maxiter": 16, "checkres": False})
    ),
    "coarsestsolve": lambda mat: s.inv_direct(
        mat, a.bicgstab({"eps": 5e-2, "maxiter": 50, "checkres": False}),
    ),
    "wrappersolve": None,
    "setupsolve": lambda mat: s.inv_eo_ne(
        p.eo2(mat), a.cg({"eps": 1e-3, "maxiter": 50, "checkres": False})
    ),
    "distribution": rng.cnormal,
}
mg_params_3lvl = {
    "grid": [
        grid_f,
        g.block.grid(grid_f, [2, 2, 2, 2]),  # 2**4 blocking
        g.block.grid(grid_f, [4, 4, 4, 4]),  # additional 2**4 -> 4**4 blocking
    ],
    "northo": 1,
    "nbasis": 40,
    "hermitian": False,
    "savelinks": True,
    "vecstype": "test",
    "presmooth": None,
    "postsmooth": [
        lambda mat: s.inv_eo(
            p.eo2(mat), a.bicgstab({"eps": 1e-1, "maxiter": 16, "checkres": False})
        ),
        lambda mat: s.inv_direct(
            mat, a.bicgstab({"eps": 1e-1, "maxiter": 16, "checkres": False})
        ),
    ],
    "coarsestsolve": lambda mat: s.inv_direct(
        mat, a.bicgstab({"eps": 5e-2, "maxiter": 50, "checkres": False}),
    ),
    "wrappersolve": lambda mat, prec: s.inv_direct(
        mat,
        a.fgmres(
            {
                "eps": 1e-1,
                "maxiter": 10,
                "restartlen": 5,
                "checkres": False,
                "prec": prec,
            }
        ),
    ),
    "setupsolve": [
        lambda mat: s.inv_eo_ne(
            p.eo2(mat), a.cg({"eps": 1e-3, "maxiter": 50, "checkres": False})
        ),
        lambda mat: s.inv_direct(
            mat,
            a.fgmres({"eps": 1e-3, "maxiter": 50, "restartlen": 25, "checkres": False}),
        ),
    ],
    "distribution": [rng.cnormal, rng.zn],
}
g.message("multigrid parameters 2lvl: ", mg_params_2lvl)
g.message("multigrid parameters 3lvl: ", mg_params_3lvl)

mg_prec_2lvl = a.mg(w_dp, mg_params_2lvl)
mg_prec_3lvl = a.mg(w_dp, mg_params_3lvl)
bicgstab_prec = a.bicgstab({"eps": 1e-1, "maxiter": 16, "checkres": False})
eo2_even_dp = p.eo2(w_dp, parity=g.even)

# solver params
eps = 1e-8
maxiter = 600
restartlen = 20

# solvers (using dictionaries to gain access to solver history)
slvs = {
    "cg_eone_none": {
        "inv": s.inv_eo_ne,
        "mat": p.eo2(w_dp),
        "alg": a.cg({"eps": eps, "maxiter": maxiter}),
    },
    "fgmres_direct_mg_2lvl": {
        "inv": s.inv_direct,
        "mat": w_dp,
        "alg": a.fgmres(
            {
                "eps": eps,
                "maxiter": maxiter,
                "restartlen": restartlen,
                "prec": s.inv_direct(w_dp, mg_prec_2lvl),
            }
        ),
    },
    "fgmres_direct_mg_3lvl": {
        "inv": s.inv_direct,
        "mat": w_dp,
        "alg": a.fgmres(
            {
                "eps": eps,
                "maxiter": maxiter,
                "restartlen": restartlen,
                "prec": s.inv_direct(w_dp, mg_prec_3lvl),
            }
        ),
    },
    "fgmres_direct_bicgstab": {
        "inv": s.inv_direct,
        "mat": w_dp,
        "alg": a.fgmres(
            {
                "eps": eps,
                "maxiter": maxiter,
                "restartlen": restartlen,
                "prec": s.inv_direct(w_dp, bicgstab_prec),
            }
        ),
    },
}

# helper dictionaries
timings = {}
resid = {}
iters = {}
resid_cg_eone_none = {}

# ensure cg runs first
assert list(slvs.keys())[0] == "cg_eone_none"


def test_slv(slv, name):
    global dst_cg_eone_none
    g.message(f"Starting with solver {name}")
    t0 = g.time()
    dst = g.eval(s.propagator(slv["inv"](slv["mat"], slv["alg"])) * src)
    if name == "cg_eone_none":
        dst_cg_eone_none = g.copy(dst)
    t1 = g.time()
    timings[name] = t1 - t0
    resid[name] = (g.norm2(w_dp.M * dst - src) / g.norm2(src)) ** 0.5
    if name == "cg_eone_none":
        resid_cg_eone_none[name] = 0.0
    else:
        resid_cg_eone_none[name] = (
            g.norm2(dst_cg_eone_none - dst) / g.norm2(dst_cg_eone_none)
        ) ** 0.5
    iters[name] = len(slv["alg"].history)


# run solvers
for k, v in slvs.items():
    test_slv(v, k)

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
    g.message(
        "%-38s %-25g %-25d %-25g %-25g"
        % (k, v, iters[k], resid[k], resid_cg_eone_none[k])
    )

# print 2lvl mg profiling
g.message(
    "Contributions to time spent in 2lvl MG preconditioner (possibly accumulated for all mg solver instances with same preconditioner)"
)
for lvl in reversed(range(len(mg_prec_2lvl.t_setup))):
    mg_prec_2lvl.t_setup[lvl].print()
for lvl in reversed(range(len(mg_prec_2lvl.t_solve))):
    mg_prec_2lvl.t_solve[lvl].print()

# print 3lvl mg profiling
g.message(
    "Contributions to time spent in 3lvl MG preconditioner (possibly accumulated for all mg solver instances with same preconditioner)"
)
for lvl in reversed(range(len(mg_prec_3lvl.t_setup))):
    mg_prec_3lvl.t_setup[lvl].print()
for lvl in reversed(range(len(mg_prec_3lvl.t_solve))):
    mg_prec_3lvl.t_solve[lvl].print()
