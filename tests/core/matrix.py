#!/usr/bin/env python3
#
# Authors: Christoph Lehner 2020
#
import gpt as g
import numpy as np

grid_dp = g.grid([8, 4, 4, 4], g.double)
grid_sp = g.grid([8, 4, 4, 4], g.single)

# test exp/log
for grid, eps in [(grid_dp, 1e-15), (grid_sp, 1e-7)]:
    rng = g.random("test")
    m = g.mcolor(grid)
    rng.lie(m)
    m2 = g.matrix.exp(g.matrix.log(m))
    eps2 = g.norm2(m - m2) / g.norm2(m)
    g.message(f"test exp(log(M)) = M: {eps2}")
    assert eps2 < eps**2

# test inv
for grid, eps in [(grid_dp, 1e-14), (grid_sp, 1e-6)]:
    rng = g.random("test")
    m = rng.cnormal(g.mspincolor(grid))
    minv = g.matrix.inv(m)
    eye = g.lattice(m)
    eye[:] = m.otype.identity()
    eps2 = g.norm2(m * minv - eye) / (12 * grid.fsites)
    g.message(f"test M*M^-1 = 1: {eps2}")
    assert eps2 < eps**2

# test det/reunitize
for grid, eps in [(grid_dp, 1e-14), (grid_sp, 1e-6)]:
    rng = g.random("test")
    m = rng.cnormal(g.mcolor(grid))
    g.qcd.reunitize(m)
    one = g.complex(grid)
    one[:] = 1.0

    eps2 = g.norm2(g.matrix.det(m) - one) / (3 * grid.fsites)
    g.message(f"test det(reunitize(m)) = 1: {eps2}")
    assert eps2 < eps**2

    eye = g.lattice(m)
    eye[:] = np.eye(3, dtype=eye[:].dtype)
    eps2 = g.norm2(g.adj(m) * m - eye) / (3 * grid.fsites)
    g.message(f"test unitariy: {eps2}")
    assert eps2 < eps**2

    m2 = g.copy(m)
    g.qcd.reunitize(m2)
    eps2 = g.norm2(m - m2) / g.norm2(m)
    g.message(f"test reunitize is projection: {eps2}")
    assert eps2 < eps**2
