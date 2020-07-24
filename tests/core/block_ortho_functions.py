#!/usr/bin/env python3
#
# Authors: Christoph Lehner 2020
#
# Desc.: Illustrate core concepts and features
#
import gpt as g
import numpy as np
import sys

# # setup grids
# grid_f = g.grid([16, 16, 16, 16], g.double)
# grid_c = g.block.grid(grid_f, [4, 4, 4, 4])

# setup grids
grid_f = g.grid([2, 2, 2, 2], g.double)
grid_c = g.block.grid(grid_f, [2, 2, 2, 1])

# size of basis
nbasis_f = 40
nbasis_c = 20

# setup rng
rng = g.random("uiae")

# setup fields
vec_x, vec_y, vec_z = (
    g.vcomplex(grid_f, nbasis_f),
    g.vcomplex(grid_f, nbasis_f),
    g.vcomplex(grid_f, nbasis_f),
)
scal, scal_other = g.complex(grid_c), g.complex(grid_c)

# Start of tests for block inner product ######################################

# initialize fields
tmp_x = g.vcomplex([1.0] * 10 + [2] * 10 + [3] * 10 + [4] * 10, nbasis_f)
tmp_y = g.vcomplex([5.0] * 10 + [6] * 10 + [7] * 10 + [8] * 10, nbasis_f)
inner_per_site = g.adj(tmp_x) * tmp_y
assert inner_per_site.real == 700
vec_x[:] = tmp_x
vec_y[:] = tmp_y

# block inner product
g.block.innerProduct(scal, vec_x, vec_y)
g.block.innerProduct_other(scal_other, vec_x, vec_y)

# numbers for expected results
blockvol = np.prod(grid_f.ldimensions) / np.prod(grid_c.ldimensions)
per_coarse_site = inner_per_site * blockvol
coarse_norm2 = per_coarse_site ** 2 * grid_c.fsites

# report error
print(
    f"coarse_norm2 = {coarse_norm2}, g.norm2(scal) = {g.norm2(scal)}, g.norm2(scal_other) = {g.norm2(scal_other)}"
)
assert coarse_norm2 == g.norm2(scal)
assert coarse_norm2 == g.norm2(scal_other)

# Start of tests for block zaxpy ##############################################

# initialize fields
vec_x[:] = 1.0
vec_y[:] = 0.0
vec_z[:] = 0.0

for fac in [0.0, 1.0, 5.0]:
    scal[:] = fac

    # block zaxpy
    g.block.zaxpy(vec_z, scal, vec_x, vec_y)

    # report error
    print(f"g.norm2(vec_z) = {g.norm2(vec_z)}")
    assert g.norm2(vec_z) == grid_f.fsites * nbasis_f * fac ** 2

# Start of tests for block normalize ##########################################

# initialize fields
vec_x = g.vspincolor(grid_f)
vec_y = g.vcomplex(grid_f, nbasis_f)
rng.cnormal(vec_x)
rng.cnormal(vec_y)

# block normalize
g.block.normalize(grid_c, vec_x)
g.block.normalize(grid_c, vec_y)

# numbers for expected results
tol = 1e-13
err_vspincolor = grid_c.fsites - g.norm2(vec_x)
err_vcomplex = grid_c.fsites - g.norm2(vec_y)

# report error
print(f"err({vec_x.otype.__name__}) = {err_vspincolor}, tol = {tol}")
print(f"err({vec_y.otype.__name__}) = {err_vcomplex}, tol = {tol}")
assert err_vspincolor <= tol
assert err_vcomplex <= tol

# block normalize, other impl
rng.cnormal(vec_x)
rng.cnormal(vec_y)
g.block.normalize_other(grid_c, vec_x)
g.block.normalize_other(grid_c, vec_y)

# report error
print(f"err({vec_x.otype.__name__}) = {err_vspincolor}, tol = {tol}")
print(f"err({vec_y.otype.__name__}) = {err_vcomplex}, tol = {tol}")
assert err_vspincolor <= tol
assert err_vcomplex <= tol
