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

# setup rng, mute
g.default.set_verbose("random", False)
g.default.set_verbose("fgmres_performance", True)  # to get timing info at bottom
rng = g.random("test_mg")

# just run with larger volume
L = [16, 8, 16, 16]

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
p = g.qcd.fermion.preconditioner


# define transitions between grids (setup)
def find_near_null_vectors(w, cgrid):
    slv = i.fgmres(eps=1e-3, maxiter=50, restartlen=25, checkres=False)(w)
    basis = g.orthonormalize(rng.cnormal([w.vector_space[0].lattice() for i in range(15)]))
    null = g.lattice(basis[0])
    null[:] = 0
    for b in basis:
        slv(b, null)
    # TODO: apply open boundaries, e.g., in this function
    g.qcd.fermion.coarse.split_chiral(basis)
    bm = g.block.map(cgrid, basis)
    bm.orthonormalize()
    bm.check_orthogonality()
    return basis


mg_setup_2lvl = i.multi_grid_setup(block_size=[[2, 2, 2, 2]], projector=find_near_null_vectors)

mg_setup_3lvl = i.multi_grid_setup(
    block_size=[[2, 2, 2, 2], [2, 1, 1, 1]], projector=find_near_null_vectors
)

mg_setup_2lvl_dp = mg_setup_2lvl(w_dp)
mg_setup_2lvl_sp = mg_setup_2lvl(w_sp)
mg_setup_3lvl_sp = mg_setup_3lvl(w_sp)

# mg inner solvers
wrapper_solver = i.fgmres({"eps": 1e-1, "maxiter": 10, "restartlen": 5, "checkres": False})
smooth_solver = i.fgmres({"eps": 1e-14, "maxiter": 8, "restartlen": 4, "checkres": False})
coarsest_solver = i.fgmres({"eps": 5e-2, "maxiter": 50, "restartlen": 25, "checkres": False})

# mg solver/preconditioner objects
mg_2lvl_vcycle_dp = i.sequence(
    i.coarse_grid(coarsest_solver, *mg_setup_2lvl_dp[0]),
    i.calculate_residual(
        "before smoother"
    ),  # optional since it costs time but helps to tune MG solver
    smooth_solver,
    i.calculate_residual("after smoother"),  # optional
)

# For timing purposes, keep variables of solvers of various levels
smooth_solver_lvl3 = smooth_solver.modified()
smooth_solver_lvl2 = smooth_solver.modified()
coarsest_solver_lvl3 = coarsest_solver.modified()

wrapper_solver_lvl2 = wrapper_solver.modified(
    prec=i.sequence(
        i.coarse_grid(coarsest_solver_lvl3, *mg_setup_3lvl_sp[1]),
        smooth_solver_lvl3,
    )
)

mg_3lvl_kcycle_sp = i.sequence(
    i.coarse_grid(
        wrapper_solver_lvl2,
        *mg_setup_3lvl_sp[0],
    ),
    smooth_solver_lvl2,
)

# Shorter version if we do not want to create timing overview below:
# mg_3lvl_kcycle_sp = i.sequence(
#     i.coarse_grid(
#         wrapper_solver.modified(
#             prec=i.sequence(
#                 i.coarse_grid(coarsest_solver, *mg_setup_3lvl_sp[1]), smooth_solver
#             )
#         ),
#         *mg_setup_3lvl_sp[0],
#     ),
#     smooth_solver,
# )

# outer solver
fgmres_params = {"eps": 1e-6, "maxiter": 1000, "restartlen": 20}

# preconditioned inversion (using only smoother, w/o coarse grid correction)
fgmres_outer = i.fgmres(fgmres_params, prec=smooth_solver)
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
assert niter_prec_2lvl_mg_vcycle_dp <= niter_prec_smooth

# preconditioned inversion (3lvl mg -- kcycle -- mixed precision)
fgmres_outer = i.fgmres(
    fgmres_params,
    prec=i.mixed_precision(mg_3lvl_kcycle_sp, g.single, g.double),
)
sol_prec_3lvl_mg_kcycle_mp = g.eval(fgmres_outer(w_dp) * src)

eps2 = g.norm2(w_dp * sol_prec_3lvl_mg_kcycle_mp - src) / g.norm2(src)
niter_prec_3lvl_mg_kcycle_mp = len(fgmres_outer.history)
g.message("Test resid/iter fgmres + 3lvl kcycle mg mixed:", eps2, niter_prec_3lvl_mg_kcycle_mp)
assert eps2 < 1e-12
assert niter_prec_3lvl_mg_kcycle_mp < niter_prec_smooth

# show timings
for slv_name, slv in [
    ("smooth_solver_lvl3", smooth_solver_lvl3),
    ("smooth_solver_lvl2", smooth_solver_lvl2),
    ("coarsest_solver_lvl3", coarsest_solver_lvl3),
    ("wrapper_solver_lvl2", wrapper_solver_lvl2),
]:
    g.message(f"\nTimings for {slv_name}:\n{slv.timer}\n")
