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
mat_c = g.mcomplex(grid, nbasis)
vec_in_c, vec_out_c = g.vcomplex(grid, nbasis), g.vcomplex(grid, nbasis)

# initialize fields
rng.cnormal(mat_c)
rng.cnormal(vec_in_c)
vec_out_c[:] = 0

# apply the link matrix
print(g.norm2(vec_out_c))
vec_out_c @= mat_c * vec_in_c
print(g.norm2(vec_out_c))
