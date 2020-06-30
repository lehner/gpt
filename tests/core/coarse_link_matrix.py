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
nbasis = 20  # doesn't work if not in fundamentals -> TODO

# define fields
link_c = g.mcomplex(grid, nbasis)
vec_in_c, vec_out_link_c, vec_out_mat_c = (
    g.vcomplex(grid, nbasis),
    g.vcomplex(grid, nbasis),
    g.vcomplex(grid, nbasis),
)

# initialize fields
rng.cnormal(link_c)
rng.cnormal(vec_in_c)
vec_out_link_c[:] = 0

# copy into coarse operator
A = [g.copy(link_c) for _ in range(9)]
mat_c = g.qcd.fermion.coarse_operator(A, {"hermitian": 1, "level": 0,},)

# apply the link matrix
vec_out_link_c @= link_c * vec_in_c
mat_c.Mdir(
    vec_out_mat_c, vec_in_c, 0, 0
)  # exploit the self coupling link, this uses Grid

# define check tolerance
tol = 0.0

# report error
diff2 = g.norm2(vec_out_link_c - vec_out_mat_c)
assert diff2 == tol
g.message("Test passed, %e == %e" % (diff2, tol))
