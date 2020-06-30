#!/usr/bin/env python3
#
# Authors: Daniel Richtmann 2020
#          Christoph Lehner 2020
#
# Desc.: Test constructed coarse operator against naive chained application
#
import gpt as g
import numpy as np
import sys

# setup fine link fields
U = g.qcd.gauge.random(g.grid([8, 8, 8, 8], g.double), g.random("test"))

# do everything in single precision
U = g.convert(U, g.single)

# setup grids
grid_f = U[0].grid
grid_c = g.grid([2, 2, 2, 2], grid_f.precision)

# setup fine matrix
mat_f = g.qcd.fermion.wilson_clover(
    U,
    {
        "kappa": 0.13565,
        "csw_r": 2.0171,
        "csw_t": 2.0171,
        "xi_0": 1,
        "nu": 1,
        "isAnisotropic": False,
        "boundary_phases": [1.0, 1.0, 1.0, 1.0],
    },
)

# setup rng
rng = g.random("ducks_smell_funny")

# number of basis vectors
nbasis_f = 20

# number of block orthogonalization steps
northo = 2

# setup basis
basis_f = [g.vspincolor(grid_f) for _ in range(nbasis_f)]
rng.cnormal(basis_f)
g.split_chiral(basis_f)

# orthonormalize basis_f
for i in range(northo):
    g.message("Block ortho step %d" % i)
    g.block.orthonormalize(grid_c, basis_f)

# check orthogonality
iproj_c, eproj_c = g.vcomplex(grid_c, nbasis_f), g.vcomplex(grid_c, nbasis_f)
for i, v in enumerate(basis_f):
    g.block.project(iproj_c, v, basis_f)
    eproj_c[:] = 0.0
    eproj_c[:, :, :, :, i] = 1.0
    err2 = g.norm2(eproj_c - iproj_c)
    g.message("Orthogonality check error for vector %d = %e" % (i, err2))
g.message("Orthogonality check done")

# create coarse link fields
A = [g.mcomplex(grid_c, nbasis_f) for _ in range(9)]
g.coarse.create_links(A, mat_f, basis_f)

# create coarse operator from links
mat_c = g.qcd.fermion.coarse_operator(A, {"hermitian": 1, "level": 0,},)

# setup fine vectors
vec_in_f = g.lattice(basis_f[0])
vec_out_f = g.lattice(basis_f[0])
vec_in_f[:] = 0
vec_out_f[:] = 0

# setup coarse vectors
vec_in_c = g.vcomplex(grid_c, nbasis_f)
vec_out_chained_c = g.vcomplex(grid_c, nbasis_f)
vec_out_constructed_c = g.vcomplex(grid_c, nbasis_f)
rng.cnormal(vec_in_c)
vec_out_chained_c[:] = 0
vec_out_constructed_c[:] = 0

# apply chained and constructed coarse operator
dt_chained, dt_constructed = 0.0, 0.0
dt_chained -= g.time()
g.block.promote(vec_in_c, vec_in_f, basis_f)
mat_f.M(vec_out_f, vec_in_f)
g.block.project(vec_out_chained_c, vec_out_f, basis_f)
dt_chained += g.time()
dt_constructed -= g.time()
mat_c.M(vec_out_constructed_c, vec_in_c)
dt_constructed += g.time()

g.message("Timings: chained = %e, constructed = %e" % (dt_chained, dt_constructed))

# define check tolerance
tol = 1e-26 if grid_f.precision == g.double else 1e-13

# report error
err2 = g.norm2(vec_out_chained_c - vec_out_constructed_c) / g.norm2(vec_out_chained_c)
g.message("Relative deviation of constructed from chained operator = %e" % err2)
assert err2 <= tol
g.message("Test passed, %e <= %e" % (err2, tol))
