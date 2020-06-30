#!/usr/bin/env python3
#
# Authors: Daniel Richtmann 2020
#          Christoph Lehner 2020
#
# Desc.: Test constructed coarse coarse operator against naive chained application
#
import gpt as g
import numpy as np
import sys

# setup grids
grid_c = g.grid([4, 4, 4, 4], g.single)
grid_cc = g.grid([2, 2, 2, 2], grid_c.precision)

# setup rng
rng = g.random("ducks_smell_funny")

# number of basis vectors
nbasis_f = 20
nbasis_c = nbasis_f

# number of block orthogonalization steps
northo = 2

# create coarse link fields
A_c = [g.mcomplex(grid_c, nbasis_f) for _ in range(9)]
g.coarse.create_links_random(A_c)

# create coarse operator from links
mat_c = g.qcd.fermion.coarse_operator(A_c, {"hermitian": 1, "level": 0,},)

# setup coarse basis
basis_c = [g.vcomplex(grid_c, nbasis_f) for _ in range(nbasis_c)]
rng.cnormal(basis_c)
# g.split_chiral(basis_c)  # doesn't work because can't apply gamma to coarse vec -> TODO

# orthonormalize coarse basis
for i in range(northo):
    g.message("Block ortho step %d" % i)
    g.block.orthonormalize(grid_cc, basis_c)

# check orthogonality
iproj_cc, eproj_cc = (
    g.vcomplex(grid_cc, nbasis_c),
    g.vcomplex(grid_cc, nbasis_c),
)
for i, v in enumerate(basis_c):
    g.block.project(iproj_cc, v, basis_c)
    eproj_cc[:] = 0.0
    eproj_cc[:, :, :, :, i] = 1.0
    err2 = g.norm2(eproj_cc - iproj_cc)
    g.message("Orthogonality check error for vector %d = %e" % (i, err2))
g.message("Orthogonality check done")

# create coarse coarse link fields
A_cc = [g.mcomplex(grid_cc, nbasis_c) for _ in range(9)]
g.coarse.create_links(
    A_cc, mat_c, basis_c
)  # doesn't work because can't mul g.complex x g.vcomplex -> TODO

# create coarse operator from links
mat_cc = g.qcd.fermion.coarse_operator(A_cc, {"hermitian": 1, "level": 1,},)

# setup coarse vectors
vec_in_c = g.lattice(basis_c[0])
vec_out_c = g.lattice(basis_c[0])
vec_in_c[:] = 0
vec_out_c[:] = 0

# setup coarse coarse vectors
vec_in_cc = g.vcomplex(grid_cc, nbasis_c)
vec_out_chained_cc = g.vcomplex(grid_cc, nbasis_c)
vec_out_constructed_cc = g.vcomplex(grid_cc, nbasis_c)
rng.cnormal(vec_in_cc)
vec_out_chained_cc[:] = 0
vec_out_constructed_cc[:] = 0

# apply chained and constructed coarse operator
dt_chained, dt_constructed = 0.0, 0.0
dt_chained -= g.time()
g.block.promote(vec_in_cc, vec_in_c, basis_c)
fmat.M(vec_out_c, vec_in_c)
g.block.project(vec_out_chained_cc, vec_out_c, basis_c)
dt_chained += g.time()
dt_constructed -= g.time()
mat_c.M(vec_out_constructed_cc, vec_in_cc)
dt_constructed += g.time()

g.message("Timings: chained = %e, constructed = %e" % (dt_chained, dt_constructed))

# define check tolerance
tol = 1e-26 if grid_c.precision == g.double else 1e-13

# report error
err2 = g.norm2(vec_out_chained_cc - vec_out_constructed_cc) / g.norm2(
    vec_out_chained_cc
)
g.message("Relative deviation of constructed from chained operator = %e" % err2)
assert err2 <= tol
g.message("Test passed, %e <= %e" % (err2, tol))
