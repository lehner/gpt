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
U0 = g.u1(grid)
V0 = g.u1(grid)
rng.element([U0, V0])

U_ref = g.u1(grid)
V_ref = g.u1(grid)
rng.element([U_ref, V_ref])


class test_functional(differentiable_functional):
    def __call__(self, fields):
        U, V = fields
        return g.norm2(U - U_ref) + g.norm2(V - V_ref)

    def deriv(self, f, f_ref):
        x = g.component.real(f)
        x_ref = g.component.real(f_ref)
        y = g.component.imag(f)
        y_ref = g.component.imag(f_ref)
        return g(2.0 * x_ref * y - 2.0 * y_ref * x)

    def gradient(self, fields, dfields):
        U, V = fields
        a = []
        for f in dfields:
            if f is U:
                r = self.deriv(U, U_ref)
            elif f is V:
                r = self.deriv(V, V_ref)
            else:
                assert False
            r.otype = V.otype.cartesian()
            a.append(r)
        return a


f = test_functional()

# first establish correctness of df
f.assert_gradient_error(rng, [U0, V0], [U0], 1e-4, 1e-10)

# now test minimizers
fr = g.algorithms.optimize.fletcher_reeves
pr = g.algorithms.optimize.polak_ribiere
ls0 = g.algorithms.optimize.line_search_none
ls2 = g.algorithms.optimize.line_search_quadratic
for gd in [
    g.algorithms.optimize.gradient_descent(maxiter=40, eps=1e-7, step=1e-1, line_search=ls0),
    g.algorithms.optimize.gradient_descent(maxiter=40, eps=1e-7, step=1e-1, line_search=ls2),
    g.algorithms.optimize.non_linear_cg(maxiter=40, eps=1e-7, step=1e-1, line_search=ls0, beta=fr),
    g.algorithms.optimize.non_linear_cg(maxiter=40, eps=1e-7, step=1e-1, line_search=ls2, beta=fr),
    g.algorithms.optimize.non_linear_cg(maxiter=40, eps=1e-7, step=1e-1, line_search=ls2, beta=pr),
    g.algorithms.optimize.adam(
        maxiter=40, eps=1e-7, alpha=1e-1, beta1=0.05, beta2=0.99, eps_regulator=0.1
    ),
]:
    U1, V1 = g.copy([U0, V0])
    assert f([U1, V1]) > 1e2
    gd(f)([U1, V1], [V1])
    assert (f([U1, V1]) - 83.827385931) < 1e-5

    U1, V1 = g.copy([U0, V0])
    assert f([U1, V1]) > 1e2
    gd(f)([U1, V1], [U1, V1])
    assert f([U1, V1]) < 1e-5

# test symmetric update functional
s = g.algorithms.group.symmetric_functional(f)
U1 = g.copy(U0)
V1 = g.copy(V1)
rng.element([U1, V1])
s.assert_gradient_error(rng, [U0, V0, U1, V1], [U0, U1], 1e-4, 1e-10)

# test locally coherent functional
cgrid = g.grid([4, 4, 4, 4], g.double)
block = g.block.transfer(grid, cgrid, U0.otype)
lc = g.algorithms.group.locally_coherent_functional(f, block)
U1 = g.lattice(cgrid, U0.otype)
V1 = g.lattice(cgrid, V0.otype)
rng.element([U1, V1])
lc.assert_gradient_error(rng, [U0, V0, U1, V1], [U0, U1], 1e-4, 1e-10)

# test polar decomposition functional
r = g.algorithms.group.polar_regulator(1.3, 0.6, 2)
f = g.qcd.gauge.action.wilson(5.3)
s = g.algorithms.group.polar_decomposition_functional(f, r)
W = [g.matrix_color_complex_additive(grid, 3) for _ in range(4)]
rng.element(W)
for w in W:
    w += g.identity(w) * 2
s.assert_gradient_error(rng, W, W, 1e-4, 1e-8)
