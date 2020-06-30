#!/usr/bin/env python3
#
# Authors: Daniel Richtmann 2020
#          Christoph Lehner 2020
#
# Desc.: Check correctness of blocking routine for coarsening
#
import gpt as g
import numpy as np

# define grids
grid_f = g.grid([8, 8, 8, 8], g.double)
grid_c = g.grid([4, 4, 4, 4], grid_f.precision)

# define test tolerance
tol = 1e-15 if grid_f.precision == g.double else 1e-7

# setup rng
rng = g.random("ducks_smell_funny")

# size of basis
nbasis_f = 40

# setup basis
basis_f = [g.vspincolor(grid_f) for i in range(nbasis_f)]
rng.zn(basis_f)
g.split_chiral(basis_f)
g.block.orthonormalize(grid_c, basis_f)

# setup fields
vec_in_orig_f, vec_in_masked_f = g.vspincolor(grid_f), g.vspincolor(grid_f)
vec_out_project_c, vec_out_mip_c = (
    g.vcomplex(grid_c, nbasis_f),
    g.vcomplex(grid_c, nbasis_f),
)
fullmask, blockmaskeven = g.complex(grid_f), g.complex(grid_f)
scalar_tmp_c = g.complex(grid_c)

# initialize fields
rng.cnormal(vec_in_orig_f)
vec_out_project_c[:] = 0
vec_out_mip_c[:] = 0
scalar_tmp_c[:] = 0

# setup mask
fullmask[:] = 1
coor = g.coordinates(blockmaskeven)
block = np.array(grid_f.ldimensions) / np.array(grid_c.ldimensions)
block_cb = coor[:, :] // block[:]
g.make_mask(blockmaskeven, np.sum(block_cb, axis=1) % 2 == 0)

# two ways of masked projections
mask = blockmaskeven
vec_in_masked_f @= mask * vec_in_orig_f
g.block.project(vec_out_project_c, vec_in_masked_f, basis_f)
for i in range(nbasis_f):
    g.block.maskedInnerProduct(scalar_tmp_c, mask, basis_f[i], vec_in_orig_f)
    vec_out_mip_c[:, :, :, :, i] = scalar_tmp_c[:]

# compare results
diff2 = g.norm2(vec_out_project_c - vec_out_mip_c)
assert diff2 <= tol
g.message("Test passed, %e <= %e" % (diff2, tol))
