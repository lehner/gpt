#!/usr/bin/env python3
#
# Authors: Daniel Richtmann 2020
#          Christoph Lehner 2020
#
# Desc.: Test different versions of constructing coarse operator against each other
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
        "mass": -0.1,
        "csw_r": 0,
        "csw_t": 0,
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

# split basis into chiral halfs
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

# define check tolerance
tol = 1e-26 if grid_f.precision == g.double else 1e-13

# create coarse link fields with and without dirsave
A_notsaved = [g.mcomplex(grid_c, nbasis_f) for __ in range(9)]
A_saved = [g.mcomplex(grid_c, nbasis_f) for __ in range(9)]
g.coarse.create_links(
    A_notsaved, mat_f, basis_f, {"hermitian": False, "savelinks": False}
)
g.coarse.create_links(A_saved, mat_f, basis_f, {"hermitian": False, "savelinks": True})

# do comparison
for p in range(9):
    err2 = g.norm2(A_notsaved[p] - A_saved[p]) / g.norm2(A_saved[p])
    g.message(f"Relative deviation of A_saved[{p}] from A_notsaved[{p}] = {err2:e}",)
    assert err2 <= tol
g.message(f"Tests passed for all directions")
