#!/usr/bin/env python3
#
# Authors: Daniel Richtmann 2020
#          Christoph Lehner 2020
#
# Desc.: Test multigrid for clover
#
import gpt as g
import numpy as np

# setup rng, mute
g.default.set_verbose("random", False)
rng = g.random("test_mg")

# just run with larger volume
L = [8, 8, 16, 16]

# setup gauge field
U = g.qcd.gauge.random(g.grid(L, g.double), rng)
g.message("Plaquette:", g.qcd.gauge.plaquette(U))

# quark
w_dp = g.qcd.fermion.wilson_clover(
    U,
    {
        "kappa": 0.137,
        "csw_r": 0,
        "csw_t": 0,
        "xi_0": 1,
        "nu": 1,
        "isAnisotropic": False,
        "boundary_phases": [1.0, 1.0, 1.0, 1.0],
    },
)
w_sp = w_dp.converted(g.single)

# default grid
grid = U[0].grid

# create source
src = g.vspincolor(grid)
src[:] = g.vspincolor([[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]])

# abbreviations
i = g.algorithms.inverter
mg = i.multi_grid
p = g.qcd.fermion.preconditioner

# NOTE: mg params structure
# - list -> configure each level by itself explicitly (correct lengths asserted inside)
# - scalar value (= not a list) -> broadcast parameter to every level

# mg setup parameters
mg_setup_2lvl_params = {
    "block": [[2, 2, 2, 2]],
    "northo": 1,
    "nbasis": 30,
    "make_hermitian": False,
    "save_links": True,
    "vecstype": "null",
    "preortho": False,
    "postortho": False,
    "solver": i.fgmres(
        {"eps": 1e-3, "maxiter": 50, "restartlen": 25, "checkres": False}
    ),
    "distribution": rng.cnormal,
}
mg_setup_3lvl_params = {
    "block": [[2, 2, 2, 2], [1, 2, 2, 2]],
    "northo": 1,
    "nbasis": 30,
    "make_hermitian": False,
    "save_links": True,
    "vecstype": "null",
    "preortho": False,
    "postortho": False,
    "solver": i.fgmres(
        {"eps": 1e-3, "maxiter": 50, "restartlen": 25, "checkres": False}
    ),
    "distribution": rng.cnormal,
}
mg_setup_4lvl_params = {
    "block": [[2, 2, 2, 2], [1, 2, 1, 1], [1, 1, 2, 2]],
    "northo": 1,
    "nbasis": 30,
    "make_hermitian": False,
    "save_links": True,
    "vecstype": "null",
    "preortho": False,
    "postortho": False,
    "solver": i.fgmres(
        {"eps": 1e-3, "maxiter": 50, "restartlen": 25, "checkres": False}
    ),
    "distribution": rng.cnormal,
}
g.message(f"mg_setup_2lvl = {mg_setup_2lvl_params}")
g.message(f"mg_setup_3lvl = {mg_setup_3lvl_params}")
g.message(f"mg_setup_4lvl = {mg_setup_4lvl_params}")

# mg setup objects
mg_setup_2lvl_dp = mg.setup(w_dp, mg_setup_2lvl_params)
mg_setup_2lvl_sp = mg.setup(w_sp, mg_setup_2lvl_params)
mg_setup_3lvl_sp = mg.setup(w_sp, mg_setup_3lvl_params)
# mg_setup_4lvl_sp = mg.setup(w_sp, mg_setup_4lvl_params)

# mg inner solvers
wrapper_solver = i.fgmres(
    {"eps": 1e-1, "maxiter": 10, "restartlen": 5, "checkres": False}
)
smooth_solver = i.fgmres(
    {"eps": 1e-14, "maxiter": 8, "restartlen": 4, "checkres": False}
)
coarsest_solver = i.fgmres(
    {"eps": 5e-2, "maxiter": 50, "restartlen": 25, "checkres": False}
)

# mg solver/preconditioner objects
vcycle_params = {
    "coarsest_solver": coarsest_solver,
    "smooth_solver": smooth_solver,
    "wrapper_solver": None,
}
kcycle_params = {
    "coarsest_solver": coarsest_solver,
    "smooth_solver": smooth_solver,
    "wrapper_solver": wrapper_solver,
}
mg_2lvl_vcycle_dp = mg.inverter(mg_setup_2lvl_dp, vcycle_params)
mg_2lvl_vcycle_sp = mg.inverter(mg_setup_2lvl_sp, vcycle_params)
# mg_2lvl_kcycle_sp = mg.inverter(mg_setup_2lvl_sp, kcycle_params)
# mg_3lvl_vcycle_sp = mg.inverter(mg_setup_3lvl_sp, vcycle_params)
mg_3lvl_kcycle_sp = mg.inverter(mg_setup_3lvl_sp, kcycle_params)
# mg_4lvl_vcycle_sp = mg.inverter(mg_setup_4lvl_sp, vcycle_params)
# mg_4lvl_kcycle_sp = mg.inverter(mg_setup_4lvl_sp, kcycle_params)

# preconditioners
smoother_prec = mg_3lvl_kcycle_sp.smooth_solver[0]

# outer solver
fgmres_params = {"eps": 1e-6, "maxiter": 1000, "restartlen": 20}

# preconditioned inversion (using only smoother, w/o coarse grid correction)
fgmres_outer = i.fgmres(fgmres_params, prec=smoother_prec)
sol_smooth = g.eval(fgmres_outer(w_dp) * src)
eps2 = g.norm2(w_dp * sol_smooth - src) / g.norm2(src)
niter_prec_smooth = len(fgmres_outer.history)
g.message("Test resid/iter fgmres + smoother:", eps2, niter_prec_smooth)
assert eps2 < 1e-12

# preconditioned inversion (2lvl mg -- vcycle -- double precision)
fgmres_outer = i.fgmres(fgmres_params, prec=mg_2lvl_vcycle_dp)
sol_prec_2lvl_mg_vcycle_dp = g.eval(fgmres_outer(w_dp) * src)
eps2 = g.norm2(w_dp * sol_prec_2lvl_mg_vcycle_dp - src) / g.norm2(src)
niter_prec_2lvl_mg_vcycle_dp = len(fgmres_outer.history)
g.message(
    "Test resid/iter fgmres + 2lvl vcycle mg double:",
    eps2,
    niter_prec_2lvl_mg_vcycle_dp,
)
assert eps2 < 1e-12
assert niter_prec_2lvl_mg_vcycle_dp < niter_prec_smooth

# preconditioned inversion (2lvl mg -- vcycle -- mixed precision)
fgmres_outer = i.fgmres(
    fgmres_params,
    prec=i.mixed_precision(i.direct(mg_2lvl_vcycle_sp), g.single, g.double),
    # prec=i.mixed_precision(mg_2lvl_vcycle_sp, g.single, g.double),  # NOTE: This won't work
)
sol_prec_2lvl_mg_vcycle_mp = g.eval(fgmres_outer(w_dp) * src)
eps2 = g.norm2(w_dp * sol_prec_2lvl_mg_vcycle_mp - src) / g.norm2(src)
niter_prec_2lvl_mg_vcycle_mp = len(fgmres_outer.history)
g.message(
    "Test resid/iter fgmres + 2lvl vcycle mg mixed:", eps2, niter_prec_2lvl_mg_vcycle_mp
)
assert eps2 < 1e-12
assert niter_prec_2lvl_mg_vcycle_mp <= niter_prec_2lvl_mg_vcycle_dp + 1

# # preconditioned inversion (2lvl mg -- kcycle -- mixed precision)
# fgmres_outer = i.fgmres(
#     fgmres_params,
#     prec=i.mixed_precision(i.direct(mg_2lvl_kcycle_sp), g.single, g.double),
# )
# sol_prec_2lvl_mg_kcycle_mp = g.eval(fgmres_outer(w_dp) * src)
# eps2 = g.norm2(w_dp * sol_prec_2lvl_mg_kcycle_mp - src) / g.norm2(src)
# niter_prec_2lvl_mg_kcycle_mp = len(fgmres_outer.history)
# g.message(
#     "Test resid/iter fgmres + 2lvl kcycle mg mixed:", eps2, niter_prec_2lvl_mg_kcycle_mp
# )
# assert eps2 < 1e-12
# assert (
#     niter_prec_2lvl_mg_kcycle_mp == niter_prec_2lvl_mg_vcycle_mp
# )  # equivalent for 2 lvls

# # preconditioned inversion (3lvl mg -- vcycle -- mixed precision)
# fgmres_outer = i.fgmres(
#     fgmres_params,
#     prec=i.mixed_precision(i.direct(mg_3lvl_vcycle_sp), g.single, g.double),
# )
# sol_prec_3lvl_mg_vcycle_mp = g.eval(fgmres_outer(w_dp) * src)
# eps2 = g.norm2(w_dp * sol_prec_3lvl_mg_vcycle_mp - src) / g.norm2(src)
# niter_prec_3lvl_mg_vcycle_mp = len(fgmres_outer.history)
# g.message(
#     "Test resid/iter fgmres + 3lvl vcycle mg mixed:", eps2, niter_prec_3lvl_mg_vcycle_mp
# )
# assert eps2 < 1e-12
# assert niter_prec_3lvl_mg_vcycle_mp < niter_prec_smooth

# preconditioned inversion (3lvl mg -- kcycle -- mixed precision)
fgmres_outer = i.fgmres(
    fgmres_params,
    prec=i.mixed_precision(i.direct(mg_3lvl_kcycle_sp), g.single, g.double),
)
sol_prec_3lvl_mg_kcycle_mp = g.eval(fgmres_outer(w_dp) * src)
eps2 = g.norm2(w_dp * sol_prec_3lvl_mg_kcycle_mp - src) / g.norm2(src)
niter_prec_3lvl_mg_kcycle_mp = len(fgmres_outer.history)
g.message(
    "Test resid/iter fgmres + 3lvl kcycle mg mixed:", eps2, niter_prec_3lvl_mg_kcycle_mp
)
assert eps2 < 1e-12
assert niter_prec_3lvl_mg_kcycle_mp <= niter_prec_3lvl_mg_vcycle_mp

# # preconditioned inversion (4lvl mg -- vcycle -- mixed precision)
# fgmres_outer = i.fgmres(
#     fgmres_params,
#     prec=i.mixed_precision(i.direct(mg_4lvl_vcycle_sp), g.single, g.double),
# )
# sol_prec_4lvl_mg_vcycle_mp = g.eval(fgmres_outer(w_dp) * src)
# eps2 = g.norm2(w_dp * sol_prec_4lvl_mg_vcycle_mp - src) / g.norm2(src)
# niter_prec_4lvl_mg_vcycle_mp = len(fgmres_outer.history)
# g.message(
#     "Test resid/iter fgmres + 4lvl vcycle mg mixed:", eps2, niter_prec_4lvl_mg_vcycle_mp
# )
# assert eps2 < 1e-12
# assert niter_prec_4lvl_mg_vcycle_mp <= niter_prec_3lvl_mg_vcycle_mp

# # preconditioned inversion (4lvl mg -- kcycle -- mixed precision)
# fgmres_outer = i.fgmres(fgmres_params, prec=mg_4lvl_kcycle_sp)
# fgmres_outer = i.fgmres(
#     fgmres_params,
#     prec=i.mixed_precision(i.direct(mg_4lvl_kcycle_sp), g.single, g.double),
# )
# sol_prec_4lvl_mg_kcycle_mp = g.eval(fgmres_outer(w_dp) * src)
# eps2 = g.norm2(w_dp * sol_prec_4lvl_mg_kcycle_mp - src) / g.norm2(src)
# niter_prec_4lvl_mg_kcycle_mp = len(fgmres_outer.history)
# g.message(
#     "Test resid/iter fgmres + 4lvl kcycle mg mixed:", eps2, niter_prec_4lvl_mg_kcycle_mp
# )
# assert eps2 < 1e-12
# assert niter_prec_4lvl_mg_kcycle_mp <= niter_prec_4lvl_mg_vcycle_mp

# print contributions to mg setup runtime
g.message("Contributions to time spent in MG setups")
for name, t in [
    ("2lvl_dp", mg_setup_2lvl_dp.t),
    ("2lvl_sp", mg_setup_2lvl_sp.t),
    ("3lvl_sp", mg_setup_3lvl_sp.t),
    # ("4lvl_sp", mg_setup_4lvl_sp.t),
]:
    g.message(name + ":")
    for lvl in reversed(range(len(t))):
        g.message(t[lvl])

# print contributions to mg solve runtime
g.message("Contributions to time spent in MG preconditioners")
for name, t in [
    ("2lvl_vcycle_dp", mg_2lvl_vcycle_dp.t),
    ("2lvl_vcycle_sp", mg_2lvl_vcycle_sp.t),
    # ("2lvl_kcycle_sp", mg_2lvl_kcycle_sp.t),
    # ("3lvl_vcycle_sp", mg_3lvl_vcycle_sp.t),
    ("3lvl_kcycle_sp", mg_3lvl_kcycle_sp.t),
    # ("4lvl_vcycle_sp", mg_4lvl_vcycle_sp.t),
    # ("4lvl_kcycle_sp", mg_4lvl_kcycle_sp.t),
]:
    g.message(name + ":")
    for lvl in reversed(range(len(t))):
        g.message(t[lvl])

# print average iteration counts / time per level
g.message("Average iteration counts of inner solvers")
for name, h in [
    ("2lvl_vcycle_dp", mg_2lvl_vcycle_dp.history),
    ("2lvl_vcycle_sp", mg_2lvl_vcycle_sp.history),
    # ("2lvl_kcycle_sp", mg_2lvl_kcycle_sp.history),
    # ("3lvl_vcycle_sp", mg_3lvl_vcycle_sp.history),
    ("3lvl_kcycle_sp", mg_3lvl_kcycle_sp.history),
    # ("4lvl_vcycle_sp", mg_4lvl_vcycle_sp.history),
    # ("4lvl_kcycle_sp", mg_4lvl_kcycle_sp.history),
]:
    for lvl in reversed(range(len(h))):
        for k, v in h[lvl].items():
            stats = list(map(lambda l: sum(l) / len(l), zip(*v)))
            if stats:
                g.message(f"{name}: lvl {lvl}: {k:10s} = {int(stats[0])}")
