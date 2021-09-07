#!/usr/bin/env python3
#
# Author: Christoph Lehner 2021
#
import gpt as g
import sys

# Parameters
p_mpi_split = g.default.get_ivec("--mpi_split", None, 3)
p_maxiter_cg = g.default.get_int("--maxiter_cg", 500)
p_maxiter_gd = g.default.get_int("--maxiter_gd", 100000)
p_eps = g.default.get_float("--eps", 3e-8)
p_step = g.default.get_float("--step", 0.03)
p_source = g.default.get("--source", None)
p_rng_seed = g.default.get("--random", None)

g.message(
    f"""

  Coulomb gauge fixer run with:

    maxiter_cg  = {p_maxiter_cg}
    maxiter_gd  = {p_maxiter_gd}
    eps         = {p_eps}
    step        = {p_step}
    source      = {p_source}
    random      = {p_rng_seed}

  Note: convergence is only guaranteed for sufficiently small step parameter.

"""
)

if p_source is None:
    g.message("Need to provide source file")
    sys.exit(1)

if p_mpi_split is None:
    g.message("Need to provide mpi_split")
    sys.exit(1)

# create rng if needed
rng = None if p_rng_seed is None else g.random(p_rng_seed)

# load source
U = g.load(p_source)

# split in time
Nt = U[0].grid.gdimensions[3]
g.message(f"Separate {Nt} time slices")
Usep = [g.separate(u, 3) for u in U[0:3]]
Vt = [g.mcolor(Usep[0][0].grid) for t in range(Nt)]
cache = {}
split_grid = Usep[0][0].grid.split(p_mpi_split, Usep[0][0].grid.fdimensions)

g.message("Split grid")
Usep_split = [g.split(Usep[mu], split_grid, cache) for mu in range(3)]
Vt_split = g.split(Vt, split_grid, cache)

# optimizer
opt = g.algorithms.optimize
cg = opt.non_linear_cg(
    maxiter=p_maxiter_cg,
    eps=p_eps,
    step=p_step,
    line_search=opt.line_search_quadratic,
    beta=opt.polak_ribiere,
)
gd = opt.gradient_descent(maxiter=p_maxiter_gd, eps=p_eps, step=p_step)

# Coulomb functional on each time-slice
Nt_split = len(Vt_split)
g.message(f"This rank has {Nt_split} time slices")
for t in range(Nt_split):

    f = g.qcd.gauge.fix.landau([Usep_split[mu][t] for mu in range(3)])
    fa = opt.fourier_accelerate.inverse_phat_square(Vt_split[t].grid, f)

    g.message(f"Run local time slice {t} / {Nt_split}")

    if rng is not None:
        rng.element(Vt_split[t])
    else:
        Vt_split[t] @= g.identity(Vt_split[t])

    if not cg(fa)(Vt_split[t], Vt_split[t]):
        gd(fa)(Vt_split[t], Vt_split[t])

    group_defect = g.group.defect(Vt_split[t])
    g.message(f"Distance to group manifold: {group_defect}")
    assert group_defect < 1e-12

g.message("Unsplit")

g.unsplit(Vt, Vt_split, cache)

g.message("Project to group (should only remove rounding errors)")

Vt = [g.project(vt, "defect") for vt in Vt]

g.message("Test")

# test results
for t in range(Nt):
    f = g.qcd.gauge.fix.landau([Usep[mu][t] for mu in range(3)])
    dfv = f.gradient(Vt[t], Vt[t])
    theta = g.norm2(dfv).real / Vt[t].grid.gsites / dfv.otype.Nc
    g.message(f"theta[{t}] = {theta}")
    g.message(f"V[{t}][0,0,0] = ", Vt[t][0, 0, 0])

# merge time slices
V = g.merge(Vt, 3)
U_transformed = g.qcd.gauge.transformed(U, V)

# remove rounding errors on U_transformed
U_transformed = [g.project(u, "defect") for u in U_transformed]

# save results
g.save("U.transformed", U_transformed, g.format.nersc())
g.save("V", V)
