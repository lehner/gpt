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
    basis = g.orthonormalize(rng.cnormal([w.vector_space[0].lattice() for i in range(30)]))
    null = g.lattice(basis[0])
    null[:] = 0
    for b in basis:
        slv(b, null)
    bm = g.block.map(cgrid, basis)
    for _ in range(2):
        bm.orthonormalize()
    bm.check_orthogonality()
    return bm


# ##############
# # DWF HDCG illustration

# dwf_dp = g.qcd.fermion.mobius(U, {
#     "mass": 0.01,
#     "M5": 1.8,
#     "b": 1.5,
#     "c": 0.5,
#     "Ls": 12,
#     "boundary_phases": [1.0, 1.0, 1.0, -1.0],
# })
# dwf_sp = dwf_dp.converted(g.single)

# bm = find_near_null_vectors(p.eo2_ne()(dwf_sp).Mpc, g.block.grid(dwf_sp.F_grid_eo, [12,4,2,4,4]))
# def coarsen_operator_hdcg(bm, op):
#     return bm.coarse_operator(op) #.compile(max_point_norm=4, max_point_per_dimension=[0,1,1,1,1], tolerance=1e-3)

# cg_smooth_solver = i.cg({"eps": 1e-5, "maxiter": 20})
# block_cg = i.block_cg({"eps": 1e-5, "maxiter": 1000})

# hdcg_inner = i.mixed_precision(
#     i.sequence(
#         i.coarse_grid(block_cg, bm, coarsen_operator_hdcg),
#         cg_smooth_solver,
#     ),
#     g.single,
#     g.double
# )

# slv_hdcg = i.cg(eps=1e-8, maxiter=1000, prec=hdcg_inner)
# slv_5d = i.preconditioned(
#     p.eo2_ne(),
#     slv_hdcg,
# )

# slv_smoother_only = i.cg(eps=1e-8, maxiter=1000, prec=i.mixed_precision(
#     cg_smooth_solver,
#     g.single,
#     g.double
# ))
# slv_5d_smoother_only = i.preconditioned(
#     p.eo2_ne(),
#     slv_smoother_only,
# )

# prop = dwf_dp.propagator(slv_5d)
# g(prop * src)

# prop_smoother_only = dwf_dp.propagator(slv_5d_smoother_only)
# g(prop_smoother_only * src)

# g.message(f"HDCG lowered iteration number from {len(slv_smoother_only.history)} to {len(slv_hdcg.history)}")

# sys.exit(0)

###############
# Wilson 9pt MG solver (here minimal version to illustrate the idea)

# grid
bm = find_near_null_vectors(w_sp, g.block.grid(w_sp.F_grid, [2,2,2,2]))

def coarsen_operator(bm, w_sp):
    # TODO: save c_w_sp.points to file, load again and re-create with g.block.matrix_operator.compiled(points)
    return bm.coarse_operator(w_sp).compile(max_point_norm=1, max_point_per_dimension=[1,1,1,1])

# solvers
wrapper_solver = i.fgmres({"eps": 1e-1, "maxiter": 10, "restartlen": 5, "checkres": False})
smooth_solver = i.fgmres({"eps": 1e-14, "maxiter": 8, "restartlen": 4, "checkres": False})
coarsest_solver = i.fgmres({"eps": 5e-2, "maxiter": 50, "restartlen": 25, "checkres": False})

# mg solver/preconditioner objects
mg_2lvl_vcycle_mp = i.mixed_precision(
    i.sequence(
        i.coarse_grid(coarsest_solver, bm, coarsen_operator),
        i.calculate_residual(
            "before smoother"
        ),  # optional since it costs time but helps to tune MG solver
        smooth_solver,
        i.calculate_residual("after smoother"),  # optional
    ),
    g.single,
    g.double
)

# outer solver
fgmres_params = {"eps": 1e-6, "maxiter": 1000, "restartlen": 20}

# preconditioned inversion (using only smoother, w/o coarse grid correction)
fgmres_outer = i.fgmres(fgmres_params, prec=smooth_solver)
sol_smooth = g.eval(fgmres_outer(w_dp) * src)
eps2 = g.norm2(w_dp * sol_smooth - src) / g.norm2(src)
niter_prec_smooth = len(fgmres_outer.history)
g.message("Test resid/iter fgmres + smoother:", eps2, niter_prec_smooth)
assert eps2 < 1e-12

# preconditioned inversion (2lvl mg -- vcycle -- mixed precision)
fgmres_outer = i.fgmres(fgmres_params, prec=mg_2lvl_vcycle_mp)
sol_prec_2lvl_mg_vcycle_mp = g.eval(fgmres_outer(w_dp) * src)

eps2 = g.norm2(w_dp * sol_prec_2lvl_mg_vcycle_mp - src) / g.norm2(src)
niter_prec_2lvl_mg_vcycle_mp = len(fgmres_outer.history)
g.message(
    "Test resid/iter fgmres + 2lvl vcycle mg mixed-precision:",
    eps2,
    niter_prec_2lvl_mg_vcycle_mp,
)
assert eps2 < 1e-12
assert niter_prec_2lvl_mg_vcycle_mp <= niter_prec_smooth

g.message(f"Wilson smoother only had {niter_prec_smooth} and with MG had {niter_prec_2lvl_mg_vcycle_mp} iterations")
