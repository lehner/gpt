#!/usr/bin/env python3
#
# Authors: Christoph Lehner 2020
#
import gpt as g
import numpy as np

grid_dp = g.grid([8, 4, 4, 4], g.double)
grid_sp = g.grid([8, 4, 4, 4], g.single)

for grid, eps in [(grid_dp, 1e-15), (grid_sp, 1e-7)]:
    rng = g.random("test")
    m = g.mcolor(grid)

    # first test matrix operators
    rng.lie(m)
    m2 = g.matrix.exp(g.matrix.log(m))
    eps2 = g.norm2(m - m2) / g.norm2(m)
    g.message(eps2)
    assert eps2 < eps ** 2.0

    # then test component operators
    c = g.component

    def abs_real(x):
        return np.abs(np.real(x))

    def log_real(x):
        return np.log(np.real(x))

    def sqrt_real(x):
        return np.sqrt(np.real(x))

    def sin_real(x):
        return np.sin(np.real(x))

    def asin_real(x):
        return np.arcsin(np.real(x))

    def cos_real(x):
        return np.cos(np.real(x))

    def acos_real(x):
        return np.arccos(np.real(x))

    def inv_real(x):
        return np.real(x) ** -1.0

    def pow3p45_real(x):
        return np.real(x) ** 3.45

    for op in [
        (c.imag, np.imag),
        (c.real, np.real),
        (c.abs_real, abs_real),
        (c.exp, np.exp),
        (c.log_real, log_real),
        (c.sqrt_real, sqrt_real),
        (c.sin_real, sin_real),
        (c.asin_real, asin_real),
        (c.cos_real, cos_real),
        (c.acos_real, acos_real),
        (c.inv_real, inv_real),
        (c.pow_real(3.45), pow3p45_real),
    ]:
        a = op[0](m)[0, 0, 0, 0, 1, 2]
        b = op[1](m[0, 0, 0, 0, 1, 2])
        eps2 = (abs(a - b) / abs(a)) ** 2.0
        g.message(f"Test {op[1].__name__}: {eps2}")
        assert eps2 < eps ** 2.0
