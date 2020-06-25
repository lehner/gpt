#!/usr/bin/env python3
#
# Authors: Daniel Richtmann 2020
#          Christoph Lehner 2020
#
# Desc.: Test new data type for coarse link matrices
#
import gpt as g
import numpy as np
import sys

# setup grid
grid = g.grid([4, 4, 4, 4], g.single)

# setup rng
rng = g.random("ducks_smell_funny")

# number of basis vectors
nbasis = 20

# define fields
cmat = g.mcomplex(grid, nbasis)
cvec_in, cvec_out = g.vcomplex(grid, nbasis), g.vcomplex(grid, nbasis)

# initialize fields
rng.cnormal(cmat)
rng.cnormal(cvec_in)
cvec_out[:] = 0

# apply the link matrix
print(g.norm2(cvec_out))
cvec_out @= cmat * cvec_in
print(g.norm2(cvec_out))
