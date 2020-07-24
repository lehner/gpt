#!/usr/bin/env python3
#
# Authors: Daniel Richtmann 2020
#          Christoph Lehner 2020
#
# Desc.: Test block orthonormalize
#
import gpt as g
import numpy as np
import sys

# setup grids
fine_grid = g.grid([16, 8, 8, 16], g.single)
coarse_grid = g.grid([8, 4, 4, 8], fine_grid.precision)
coarse_coarse_grid = g.grid([4, 2, 2, 4], fine_grid.precision)

# setup rng
rng = g.random("block_seed_string_13")

# number of basis vectors
nbasis_f = 30
nbasis_c = 30

# check tolerance
ortho_tol = 1e-10
diff_tol = 1e-12

# fine basis
basis_ref_f = [g.vspincolor(fine_grid) for i in range(nbasis_f)]
rng.cnormal(basis_ref_f)
basis_virtual_f = [g.copy(v) for v in basis_ref_f]
basis_virtual_other_f = [g.copy(v) for v in basis_ref_f]

# orthonormalize
for __ in range(2):
    g.block.orthonormalize(coarse_grid, basis_ref_f)
    g.block.orthonormalize_virtual(coarse_grid, basis_virtual_f)
    g.block.orthonormalize_virtual_other(coarse_grid, basis_virtual_other_f)

# check orthogonality
g.block.check_orthogonality(coarse_grid, basis_ref_f, ortho_tol)
g.block.check_orthogonality(coarse_grid, basis_virtual_f, ortho_tol)
g.block.check_orthogonality(coarse_grid, basis_virtual_other_f, ortho_tol)

# report error
for i in range(len(basis_ref_f)):
    err2_first = g.norm2(basis_ref_f[i] - basis_virtual_f[i]) / g.norm2(basis_ref_f[i])
    err2_other = g.norm2(basis_ref_f[i] - basis_virtual_other_f[i]) / g.norm2(
        basis_ref_f[i]
    )
    g.message(
        f"Fine vector {i}, rel. deviation first impl = {err2_first}, rel. deviation other impl = {err2_other}"
    )
    assert err2_first < diff_tol
    assert err2_other < diff_tol
g.message("Tests passed for fine grid")

# coarse basis
basis_ref_c = [g.vcomplex(coarse_grid, nbasis_f) for i in range(nbasis_c)]
rng.cnormal(basis_ref_c)
basis_virtual_c = [g.copy(v) for v in basis_ref_c]
basis_virtual_other_c = [g.copy(v) for v in basis_ref_c]

# orthonormalize
for __ in range(2):
    if len(basis_ref_c[0].v_obj) == 1:
        g.block.orthonormalize(coarse_coarse_grid, basis_ref_c)
    g.block.orthonormalize_virtual(coarse_coarse_grid, basis_virtual_c)
    g.block.orthonormalize_virtual_other(coarse_coarse_grid, basis_virtual_other_c)

# check orthogonality
if len(basis_ref_c[0].v_obj) == 1:
    g.block.check_orthogonality(coarse_coarse_grid, basis_ref_c, ortho_tol)
g.block.check_orthogonality(coarse_coarse_grid, basis_virtual_c, ortho_tol)
g.block.check_orthogonality(coarse_coarse_grid, basis_virtual_other_c, ortho_tol)

# report error
if len(basis_ref_c[0].v_obj) == 1:
    for i in range(len(basis_ref_c)):
        err2_first = g.norm2(basis_ref_c[i] - basis_virtual_c[i]) / g.norm2(
            basis_ref_c[i]
        )
        err2_other = g.norm2(basis_ref_c[i] - basis_virtual_other_c[i]) / g.norm2(
            basis_ref_c[i]
        )
        g.message(
            f"Coarse vector {i}, rel. deviation first impl = {err2_first}, rel. deviation other impl = {err2_other}"
        )
        assert err2_first < diff_tol
        assert err2_other < diff_tol
    g.message("Tests passed for coarse grid")
