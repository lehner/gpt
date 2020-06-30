#!/usr/bin/env python3
#
# Authors: Daniel Richtmann 2020
#          Christoph Lehner 2020
#
# Desc.: Check correctness of chiral splitting
#
import gpt as g
import numpy as np

# define grids
grid = g.grid([8, 8, 8, 8], g.double)

# setup rng
rng = g.random("ducks_smell_funny")

# size of basis
nbasis = 40
assert nbasis % 2 == 0
nb = nbasis // 2

# setup basis
basis_ref = [g.vspincolor(grid) for i in range(nb)]
basis_split = [g.vspincolor(grid) for i in range(nbasis)]
rng.cnormal(basis_ref)
for i in range(nb):
    basis_split[i] = g.copy(basis_ref[i])

# apply function to test
g.split_chiral(basis_split)

# reverse operation
for i in range(nb):
    basis_split[i] @= basis_split[i] + basis_split[i + nb]

# test correctness
for i in range(nb):
    diff2 = g.norm2(basis_ref[i] - basis_split[i])
    assert diff2 == 0.0
    g.message("Test passed for vector %d, %e == 0.0" % (i, diff2))
