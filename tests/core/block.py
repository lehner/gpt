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
fine_grid=g.grid([8,8,8,16],g.single)
coarse_grid=g.grid([4,4,4,8],fine_grid.precision)

# basis
n=30
basis=[ g.mspincolor(fine_grid) for i in range(n) ]
rng=g.random("block_seed_string_13")
rng.cnormal(basis)
for i in range(2):
    g.message("Ortho step %d" % i)
    g.block.orthonormalize(coarse_grid,basis)

# test coarse vector
lcoarse=g.vcomplex(coarse_grid,n)
rng.cnormal(lcoarse)

# temporary fine and coarse vectors
tmpf=g.lattice(basis[0])
lcoarse2=g.lattice(lcoarse)

# coarse-to-fine-to-coarse
g.block.promote(lcoarse,tmpf,basis)
g.block.project(lcoarse2,tmpf,basis)

# report error
err2=g.norm2(lcoarse-lcoarse2) / g.norm2(lcoarse)
g.message(err2)
assert(err2 < 1e-12)

