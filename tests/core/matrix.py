#!/usr/bin/env python3
#
# Authors: Christoph Lehner 2020
#
import gpt as g
import numpy as np

grid_dp = g.grid([8, 4, 4, 4], g.double)
grid_sp = g.grid([8, 4, 4, 4], g.single)

for grid, eps in [(grid_dp, 1e-14), (grid_sp, 1e-6)]:
    rng = g.random("test")
    m = g.mcolor(grid)

    # first test matrix operators
    rng.element(m)
    m2 = g.matrix.exp(g.matrix.log(m))
    eps2 = g.norm2(m - m2) / g.norm2(m)
    g.message(f"exp(log(m)) == m: {eps2}")
    assert eps2 < eps ** 2.0

    eps2 = g.norm2(g.adj(m) - g.matrix.inv(m)) / g.norm2(m)
    g.message(f"adj(U) == inv(U): {eps2}")
    assert eps2 < eps ** 2.0

    eps2 = g.norm2(g.matrix.log(g.matrix.det(g.matrix.exp(m))) - g.trace(m)) / g.norm2(
        m
    )
    g.message(f"log(det(exp(m))) == tr(m): {eps2}")
    assert eps2 < eps ** 2.0

    # then test component operators
    c = g.component

    def inv(x):
        return x ** -1.0

    def pow3p45(x):
        return x ** 3.45

    for op in [
        (c.imag, np.imag),
        (c.real, np.real),
        (c.abs, np.abs),
        (c.exp, np.exp),
        (c.sinh, np.sinh),
        (c.cosh, np.cosh),
        (c.tanh, np.tanh),
        (c.log, np.log),
        (c.asinh, np.arcsinh),
        (c.acosh, np.arccosh),
        (c.atanh, np.arctanh),
        (c.sqrt, np.sqrt),
        (c.sin, np.sin),
        (c.asin, np.arcsin),
        (c.cos, np.cos),
        (c.acos, np.arccos),
        (c.tan, np.tan),
        (c.atan, np.arctan),
        (c.inv, inv),
        (c.pow(3.45), pow3p45),
    ]:
        a = op[0](m)[0, 0, 0, 0, 1, 2]
        b = op[1](m[0, 0, 0, 0, 1, 2])
        eps2 = (abs(a - b) / abs(a)) ** 2.0
        g.message(
            f"Test {op[1].__name__}: {a} == {b} with argument {m[0, 0, 0, 0, 1, 2]}: {eps2}"
        )
        assert eps2 < eps ** 2.0
