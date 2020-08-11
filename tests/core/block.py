#!/usr/bin/env python3
#
# Authors: Christoph Lehner 2020
#
# Desc.: Illustrate core concepts and features
#
import gpt as g
import numpy as np
import sys

# load configuration
fine_grid = g.grid([16, 8, 8, 16], g.single)
coarse_grid = g.grid([8, 4, 4, 8], fine_grid.precision)

# data types
def vsc():
    return g.vspincolor(fine_grid)


def vc12():
    return g.vcomplex(fine_grid, 12)


# masks
full_mask, blockeven_mask = g.complex(fine_grid), g.complex(fine_grid)
full_mask[:] = 1
coor = g.coordinates(blockeven_mask)
block = np.array(fine_grid.ldimensions) / np.array(coarse_grid.ldimensions)
block_cb = coor[:, :] // block[:]
g.make_mask(blockeven_mask, np.sum(block_cb, axis=1) % 2 == 0)

# lookup tables
full_lut = g.lookup_table(coarse_grid, full_mask)
blockeven_lut = g.lookup_table(coarse_grid, blockeven_mask)

# basis
n = 30
res = None
tmpf_prev = None
for dtype in [vsc, vc12]:
    g.message(f"Data type {dtype.__name__}")
    basis = [dtype() for i in range(n)]
    rng = g.random("block_seed_string_13")
    rng.cnormal(basis)
    for i in range(2):
        g.message("Ortho step %d" % i)
        g.block.orthonormalize(coarse_grid, basis)

    # test coarse vector
    lcoarse = g.vcomplex(coarse_grid, n)
    rng.cnormal(lcoarse)

    # temporary fine and coarse vectors
    tmpf = g.lattice(basis[0])
    lcoarse2 = g.lattice(lcoarse)

    # coarse-to-fine-to-coarse
    g.block.promote(lcoarse, tmpf, basis)
    g.block.project(lcoarse2, tmpf, basis)

    # report error
    err2 = g.norm2(lcoarse - lcoarse2) / g.norm2(lcoarse)
    g.message(err2)
    assert err2 < 1e-12

    # more coarse fields, randomize fine field
    scoarse = g.complex(coarse_grid)
    mcoarse = g.mcomplex(coarse_grid, n)
    mcoarse2 = g.lattice(mcoarse)
    mcoarse[:], mcoarse2[:] = 0.0, 0.0
    rng.cnormal(tmpf)

    # test lookup table project against full project
    g.block.project(lcoarse, tmpf, basis)
    g.block.project_using_lut(lcoarse2, tmpf, basis, full_lut)

    # report error
    err2 = g.norm2(lcoarse - lcoarse2) / g.norm2(lcoarse)
    g.message(err2)
    assert err2 == 0.0

    # test lookup table project against masked inner product (mask with zeros)
    for j, bv in enumerate(basis):
        g.block.masked_inner_product(scoarse, blockeven_mask, bv, tmpf)
        mcoarse[:, :, :, :, j, 0] = scoarse[:]
    g.block.project_using_lut(lcoarse, tmpf, basis, blockeven_lut)
    mcoarse2[:, :, :, :, :, 0] = lcoarse[:]

    # report error
    err2 = g.norm2(mcoarse - mcoarse2) / g.norm2(mcoarse)
    g.message(err2)
    assert err2 == 0.0
