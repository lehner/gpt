#!/usr/bin/env python3
#
# Authors: Christoph Lehner 2020
#
import gpt as g
import numpy as np
import math

grid_dp = g.grid([8, 4, 4, 4], g.double)
grid_sp = g.grid([8, 4, 4, 4], g.single)

for grid, eps in [(grid_dp, 1e-14), (grid_sp, 1e-6)]:
    rng = g.random("test")
    m = g.mcolor(grid)

    # first get matrix
    rng.element(m)

    # test
    ma = g(g.adj(m))

    # and test unitarity
    eps2 = g.norm2(g.adj(m) - g.matrix.inv(m)) / g.norm2(m)
    g.message(f"adj(U) == inv(U): {eps2}")
    assert eps2 < eps**2.0

    # then test component operators
    c = g.component

    def inv(x):
        return x**-1.0

    def pow3p45(x):
        return x**3.45

    def mod0p1(x):
        return complex(math.fmod(x.real, 0.1), math.fmod(x.imag, 0.1))

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
        (c.mod(0.1), mod0p1),
    ]:
        a = op[0](m)[0, 0, 0, 0, 1, 2]
        b = op[1](m[0, 0, 0, 0, 1, 2])
        eps2 = (abs(a - b) / abs(a)) ** 2.0
        g.message(f"Test {op[1].__name__}: {a} == {b} with argument {m[0, 0, 0, 0, 1, 2]}: {eps2}")
        assert eps2 < eps**2.0

# test inv
for grid, eps in [(grid_dp, 1e-14), (grid_sp, 1e-6)]:
    g.message(
        f"""

    Test polar.decomposition for {grid.precision.__name__}

"""
    )
    rng = g.random("test")
    W = rng.normal_element(g.matrix_color_complex_additive(grid, 3))
    H, U = g.matrix.polar.decompose(W)
    err2 = g.norm2(H * U - W) / g.norm2(W)
    g.message(f"Polar decomposition closure: {err2}")
    assert err2 < eps**2
    err2 = g.norm2(H - g.adj(H)) / g.norm2(H)
    g.message(f"Polar decomposition H: {err2}")
    assert err2 < eps**2
    err2 = g.norm2(U * g.adj(U) - g.identity(U)) / g.norm2(U)
    g.message(f"Polar decomposition U: {err2}")
    assert err2 < eps**2

    g.message(
        f"""

    Test sqrt,log,exp,det,tr for {grid.precision.__name__}

"""
    )
    for dtype in [g.mspincolor, g.mcolor, g.mspin, lambda grid: g.mcomplex(grid, 8)]:
        rng = g.random("test")
        m = rng.cnormal(dtype(grid))

        sqrt_m = g.matrix.sqrt(m)
        eps2 = g.norm2(sqrt_m * sqrt_m - m) / g.norm2(m)
        g.message(f"test sqrt(M)*sqrt(M) = M for {m.otype.__name__}: {eps2}")
        assert eps2 < eps**2

        minv = g.matrix.inv(m)
        eye = g.identity(m)
        eps2 = g.norm2(m * minv - eye) / g.norm2(eye)
        g.message(f"test M*M^-1 = 1 for {m.otype.__name__}: {eps2}")
        assert eps2 < eps**2

        eps2 = g.norm2(
            g.matrix.inv(m[0, 0, 0, 0]) * m[0, 0, 0, 0] - g.identity(m[0, 0, 0, 0])
        ) / g.norm2(g.identity(m[0, 0, 0, 0]))
        assert eps2 < eps**2 * 10

        m2 = g.matrix.exp(g.matrix.log(m))
        eps2 = g.norm2(m - m2) / g.norm2(m)
        g.message(f"exp(log(m)) == m: {eps2}")
        assert eps2 < eps**2.0 * 1e3

        # make inverse well defined
        m @= eye + 0.01 * m
        eps2 = g.norm2(g.matrix.log(g.matrix.det(g.matrix.exp(m))) - g.trace(m)) / g.norm2(m)
        g.message(f"log(det(exp(m))) == tr(m): {eps2}")
        assert eps2 < eps**2.0 * 1e3
