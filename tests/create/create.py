#!/usr/bin/env python3
#
# Authors: Christoph Lehner 2020
#
import gpt as g
import numpy as np

# test sources
rng = g.random("test")
L = [8, 8, 8, 16]
grid = g.grid(L, g.double)
c = g.create

src = g.mspincolor(grid)
c.wall.z2(src, 1, rng)
g.message("Test Z2 wall")

# simple test of correct norm
x_val = g.sum(g.trace(src * src))
x_exp = L[0] * L[1] * L[2] * 12
eps = abs(x_val - x_exp) / abs(x_val)
g.message(f"Norm test: {eps}")
assert eps < 1e-13

# test wall
test1 = rng.cnormal(g.mspincolor(grid), mu=1.0, sigma=1.0)
test2 = rng.cnormal(g.mspincolor(grid), mu=1.0, sigma=1.0)
x_val = g.sum(g.trace(src * test1 * src * test2))
tmp1 = g.lattice(test1)
tmp1[:] = 0
tmp1[:, :, :, 1] = test1[:, :, :, 1]
tmp2 = g.lattice(test2)
tmp2[:] = 0
tmp2[:, :, :, 1] = test2[:, :, :, 1]
x_exp = g.sum(g.trace(tmp1 * tmp2))
eps = abs(x_val - x_exp) / abs(x_val)
g.message(f"Random test: {eps}")
assert eps < 1e-13

# sparse test
c.sparse_grid.zn(src, [1, 2, 3, 4], [4, 4, 4, 8], rng, 2)
g.message("Test sparse Z2 grid")
x_val = g.sum(g.trace(src * test1 * src * test2))
x_exp = 0.0
for n0 in [0, 1]:
    for n1 in [0, 1]:
        for n2 in [0, 1]:
            for n3 in [0, 1]:
                pos = (n0 * 4 + 1, n1 * 4 + 2, n2 * 4 + 3, n3 * 8 + 4)
                x_exp += g.trace(test1[pos] * test2[pos])
eps = abs(x_val - x_exp) / abs(x_val)
g.message(f"Random test: {eps}")
assert eps < 1e-13

# test z3 wall
c.wall.z3(src, 1, rng)
g.message("Test Z3 wall")
test3 = rng.cnormal(g.mspincolor(grid), mu=1.0, sigma=1.0)
x_val = g.sum(g.trace(src * test1 * src * test2 * src * test3))
tmp3 = g.lattice(test1)
tmp3[:] = 0
tmp3[:, :, :, 1] = test3[:, :, :, 1]
x_exp = g.sum(g.trace(tmp1 * tmp2 * tmp3))
eps = abs(x_val - x_exp) / abs(x_val)
g.message(f"Random test: {eps}")
assert eps < 1e-13
