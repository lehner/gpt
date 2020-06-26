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
fine_grid = g.grid([8, 8, 8, 8], g.double)
coarse_grid = g.grid([4, 4, 4, 4], fine_grid.precision)

# define test tolerance
tol = 1e-15 if fine_grid.precision == g.double else 1e-7

# setup rng
rng = g.random("ducks_smell_funny")

# size of basis
nbasis = 40

# setup basis
basis = [g.vspincolor(fine_grid) for i in range(nbasis)]
rng.zn(basis)
g.split_chiral(basis)
g.block.orthonormalize(coarse_grid, basis)

# setup fields
fvec_in_orig, fvec_in_masked = g.vspincolor(fine_grid), g.vspincolor(fine_grid)
cvec_out_project, cvec_out_mip = (
    g.vcomplex(coarse_grid, nbasis),
    g.vcomplex(coarse_grid, nbasis),
)
fullmask, blockmaskeven = g.complex(fine_grid), g.complex(fine_grid)
cscalar_tmp = g.complex(coarse_grid)

# initialize fields
rng.cnormal(fvec_in_orig)
cvec_out_project[:] = 0
cvec_out_mip[:] = 0
cscalar_tmp[:] = 0

# setup mask
fullmask[:] = 1
coor = g.coordinates(blockmaskeven)
block = np.array(fine_grid.ldimensions) / np.array(coarse_grid.ldimensions)
block_cb = coor[:, :] // block[:]
g.make_mask(blockmaskeven, np.sum(block_cb, axis=1) % 2 == 0)

# two ways of masked projections
mask = blockmaskeven
fvec_in_masked @= mask * fvec_in_orig
g.block.project(cvec_out_project, fvec_in_masked, basis)
for i in range(nbasis):
    g.block.maskedInnerProduct(cscalar_tmp, mask, basis[i], fvec_in_orig)
    cvec_out_mip[:, :, :, :, i] = cscalar_tmp[:]

# compare results
diff2 = g.norm2(cvec_out_project - cvec_out_mip)
assert diff2 <= tol
g.message("Test passed, %e <= %e" % (diff2, tol))
