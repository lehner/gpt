#!/usr/bin/env python3
#
# Authors: Christoph Lehner 2021
#
import gpt as g

# load configuration
rng = g.random("test")
grid = g.grid([4, 4, 4, 8], g.double)

# test a simple functional
U = g.u1(grid)
V0 = g.u1(grid)
rng.element([U, V0])


def f(V):
    return g.sum(U * V).real ** 2.0


def df(V):
    # lim_theta->0 1/theta (g.sum(U*e^{itheta}).real ** 2.0 - g.sum(U).real ** 2.0)
    r = g(-2.0 * g.sum(U * V).real * g.component.imag(g(U * V)))
    r.otype = V.otype.cartesian()
    return r


# first establish correctness of df
df_app = g.group.approximate_gradient(V0, f, 0, 0, 0, 0)
df_val = df(V0)[0, 0, 0, 0]
eps = abs(df_app - df_val) / abs(df_val)
g.message(f"Test gradient: {eps}")
assert eps < 1e-6

# test a second functional
vp = g.complex(grid)
vp[:] = complex(1.9, 2.5)
assert (
    abs(
        g.group.approximate_gradient(vp, lambda v: g.norm2(v), 0, 0, 0, 0)
        - 2.0 * vp[0, 0, 0, 0]
    )
    < 1e-5
)

# now test minimizers
rng.element(V0)
V1 = g.copy(V0)
for gd in [
    g.algorithms.optimize.gradient_descent(maxiter=100, eps=1e-7, step=1e-3),
    g.algorithms.optimize.gradient_descent(
        maxiter=100, eps=1e-7, step=1e-3, line_search=True
    ),
    g.algorithms.optimize.non_linear_cg(
        maxiter=100, eps=1e-7, step=1e-3, line_search=False
    ),
    g.algorithms.optimize.non_linear_cg(
        maxiter=100, eps=1e-7, step=1e-3, line_search=True
    ),
]:
    V0 @= V1
    assert f(V0) > 1e3
    gd(f, df)(V0)
    assert f(V0) < 1e-7
