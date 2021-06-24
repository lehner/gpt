#!/usr/bin/env python3
#
# Authors: Christoph Lehner 2021
#
import gpt as g
from gpt.core.group import differentiable_functional

# load configuration
rng = g.random("test")
grid = g.grid([4, 4, 4, 8], g.double)

# test a simple functional
U = g.u1(grid)
V0 = g.u1(grid)
rng.element([U, V0])


class test_functional(differentiable_functional):
    def __init__(self, U):
        self.U = U

    def __call__(self, V):
        V = g.util.from_list(V)
        return g.sum(self.U * V).real ** 2.0

    @differentiable_functional.one_field_gradient
    def gradient(self, V):
        # lim_theta->0 1/theta (g.sum(U*e^{itheta}).real ** 2.0 - g.sum(U).real ** 2.0)
        r = g(-2.0 * g.sum(self.U * V).real * g.component.imag(g(self.U * V)))
        r.otype = V.otype.cartesian()
        return r


f = test_functional(U)

# first establish correctness of df
f.assert_gradient_error(rng, V0, 1e-5, 1e-8)

# now test minimizers
rng.element(V0)
V1 = g.copy(V0)
fr = g.algorithms.optimize.fletcher_reeves
pr = g.algorithms.optimize.polak_ribiere
ls0 = g.algorithms.optimize.line_search_none
ls2 = g.algorithms.optimize.line_search_quadratic
for gd in [
    g.algorithms.optimize.gradient_descent(
        maxiter=40, eps=1e-7, step=1e-3, line_search=ls0
    ),
    g.algorithms.optimize.gradient_descent(
        maxiter=40, eps=1e-7, step=1e-3, line_search=ls2
    ),
    g.algorithms.optimize.non_linear_cg(
        maxiter=40, eps=1e-7, step=1e-3, line_search=ls0, beta=fr
    ),
    g.algorithms.optimize.non_linear_cg(
        maxiter=40, eps=1e-7, step=1e-3, line_search=ls2, beta=fr
    ),
    g.algorithms.optimize.non_linear_cg(
        maxiter=40, eps=1e-7, step=1e-3, line_search=ls2, beta=pr
    ),
]:
    V0 @= V1
    assert f(V0) > 1e3
    gd(f)(V0)
    assert f(V0) < 1e-7
