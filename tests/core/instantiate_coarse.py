#!/usr/bin/env python3
#
# Authors: Daniel Richtmann 2020
#          Christoph Lehner 2020
#
# Desc.: Test instantiation of coarse operator for a high number of levels
#
import gpt as g
import numpy as np
import sys

# NOTE: We should test composite against native here!

# setup grid
grid=g.grid([4,4,4,4],g.single)

# setup rng
rng=g.random("ducks_smell_funny")

# number of basis vectors
nbasis = 20

# number of mg levels to use
# 1 = only fine, 2 = 1 cooarse level, ...
mg_levels = 18

# coarse link fields
A = [g.mcomplex(grid, nbasis) for i in range(9)]
rng.cnormal(A)

# setup coarse operators + vectors for the respective levels
num_coarse_levels = mg_levels - 1
op_c = []
vec_in_c = []
vec_out_c = []
for lvl_c in range(num_coarse_levels):
    lvl_mg = lvl_c + 1
    op_c.append(g.qcd.fermion.coarse_operator(A, {"hermitian": 1, "level": lvl_c}))
    vec_in_c.append(g.vcomplex(grid, nbasis))
    vec_out_c.append(g.vcomplex(grid, nbasis))
    rng.cnormal(vec_in_c[lvl_c])
    vec_out_c[lvl_c][:] = 0
    g.message("lvl_mg = %d, lvl_c = %d: Finished setup" % (lvl_mg, lvl_c))

# apply coarse operator
for lvl_c in range(num_coarse_levels):
    lvl_mg = lvl_c + 1
    op_c[lvl_c].M(vec_out_c[lvl_c], vec_in_c[lvl_c])
    g.message(
        "lvl_mg = %d, lvl_c = %d: in = %e" % (lvl_mg, lvl_c, g.norm2(vec_in_c[lvl_c]))
    )
    g.message(
        "lvl_mg = %d, lvl_c = %d: out = %e" % (lvl_mg, lvl_c, g.norm2(vec_out_c[lvl_c]))
    )

g.message("All tests passed")

# NOTE: This doesn't do any checks other than that instantiation and application
# work for an arbitrary number of levels
# Execution until here is success
