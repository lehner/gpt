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
def msc():
    return g.vspincolor(fine_grid)


def vc12():
    return g.vcomplex(fine_grid, 12)


# basis
n = 30
nvec = 2
res = None
tmpf_prev = None
for dtype in [msc, vc12]:
    g.message(f"Data type {dtype.__name__}")
    basis = [dtype() for i in range(n)]
    rng = g.random("block_seed_string_13")
    rng.cnormal(basis)

    b = g.block.map(coarse_grid, basis)

    for i in range(2):
        g.message("Ortho step %d" % i)
        b.orthonormalize()

    # test coarse vector
    lcoarse = [g.vcomplex(coarse_grid, n) for i in range(nvec)]
    rng.cnormal(lcoarse)

    # report error of promote-project cycle
    lcoarse2 = g( b.project * b.promote * lcoarse )
    for i in range(nvec):
        lcoarse2_i = g( b.project * b.promote * lcoarse[i] )
        eps2 = g.norm2( lcoarse2[i] - lcoarse2_i ) / g.norm2(lcoarse2_i)
        g.message(eps2)
        assert eps2 < 1e-12
    err2 = g.norm2(lcoarse2[0] - lcoarse[0]) / g.norm2(lcoarse[0])
    g.message(err2)
    assert err2 < 1e-12
