#!/usr/bin/env python3
#
# Authors: Christoph Lehner 2021
#
import gpt as g
import numpy as np
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
    g.algorithms.optimize.lbfgs(maxiter=40, eps=1e-7, step=1e-1),
]:
    U1, V1 = g.copy([U0, V0])
    assert f([U1, V1]) > 1e2
    gd(f)([U1, V1], [V1])
    assert (f([U1, V1]) - 83.827385931) < 1e-5

    U1, V1 = g.copy([U0, V0])
    assert f([U1, V1]) > 1e2
    gd(f)([U1, V1], [U1, V1])
    assert f([U1, V1]) < 1e-5

# opt.on(x, dx)(f) is opt(f)(x, dx); a run created by opt.on keeps the
# optimizer's state across calls, also with a different f in each call
for opt in [
    g.algorithms.optimize.gradient_descent(maxiter=10, eps=1e-7, step=1e-1, line_search=ls2),
    g.algorithms.optimize.non_linear_cg(maxiter=10, eps=1e-7, step=1e-1, line_search=ls2, beta=pr),
    g.algorithms.optimize.adam(maxiter=10, eps=1e-7, alpha=1e-1, beta1=0.05, beta2=0.99, eps_regulator=0.1),
]:
    U1, V1 = g.copy([U0, V0])
    U2, V2 = g.copy([U0, V0])
    opt(f)([U1, V1], [U1, V1])
    opt.on([U2, V2])(f)
    assert g.norm2(U1 - U2) == 0.0 and g.norm2(V1 - V2) == 0.0

adam = g.algorithms.optimize.adam(maxiter=40, eps=1e-15, alpha=1e-1, beta1=0.05, beta2=0.99, eps_regulator=0.1)
adam_1 = g.algorithms.optimize.adam(maxiter=1, eps=1e-15, alpha=1e-1, beta1=0.05, beta2=0.99, eps_regulator=0.1)
U1, V1 = g.copy([U0, V0])
U2, V2 = g.copy([U0, V0])
adam.on([U1, V1])(f)  # 40 steps in one call
run = adam_1.on([U2, V2])
for i in range(40):
    run(f)  # 40 calls of one step: the moments are kept
assert g.norm2(U1 - U2) == 0.0 and g.norm2(V1 - V2) == 0.0
g.message("adam: 40 steps in one call and in 40 calls agree")

# repeated calls of opt(f) keep its state as well
U3, V3 = g.copy([U0, V0])
U4, V4 = g.copy([U0, V0])
call = adam(f)
run = adam.on([U4, V4])
for i in range(3):
    call([U3, V3], [U3, V3])
    run(f)
assert g.norm2(U3 - U4) == 0.0 and g.norm2(V3 - V4) == 0.0

# a different functional in each step (here alternating f and 2 f, which have
# the same minimum) with one Adam state
U2, V2 = g.copy([U0, V0])
run = adam_1.on([U2, V2])
f2 = 2.0 * f
for i in range(40):
    run(f if i % 2 == 0 else f2)
g.message(f"adam with alternating functionals: {f([U0, V0])} -> {f([U2, V2])}")
assert f([U2, V2]) < 1e-2 * f([U0, V0])

# L-BFGS on numbers, numpy arrays and lattices (group fields: above)
class rosenbrock(differentiable_functional):
    # (1 - Re a)^2 + 100 (Re b - (Re a)^2)^2 + (Im a)^2 + (Im b)^2, minimum at a = b = 1
    def __call__(self, fields):
        a, b = fields
        return (1 - a.real) ** 2 + 100 * (b.real - a.real**2) ** 2 + a.imag**2 + b.imag**2

    def gradient(self, fields, dfields):
        # dL/dRe + i dL/dIm
        a, b = fields
        r = []
        for d in dfields:
            if d is a:
                r.append(-2 * (1 - a.real) - 400 * a.real * (b.real - a.real**2) + 2j * a.imag)
            else:
                r.append(200 * (b.real - a.real**2) + 2j * b.imag)
        return r


opt = g.algorithms.optimize.lbfgs(maxiter=200, eps=1e-10)
x = [complex(-1.2, 0.3), complex(1.0, -0.2)]
assert opt(rosenbrock())(x, x)
g.message(f"lbfgs rosenbrock: {x}")
assert abs(x[0] - 1) < 1e-8 and abs(x[1] - 1) < 1e-8

z_ref = g.complex(grid)
rng.cnormal(z_ref)
w_ref = np.array([1.0 + 2.0j, -0.5j])


class quadratic(differentiable_functional):
    # |z - z_ref|^2 + |w - w_ref|^2 (z a complex lattice, w a numpy array)
    def __call__(self, fields):
        z, w = fields
        return g.norm2(z - z_ref) + float(np.sum(np.abs(w - w_ref) ** 2))

    def gradient(self, fields, dfields):
        z, w = fields
        return [g(2.0 * (z - z_ref)) if d is z else 2.0 * (w - w_ref) for d in dfields]


z = g.complex(grid)
z[:] = 0
x = [z, np.zeros(2, dtype=np.complex128)]
q = quadratic()
q.assert_gradient_error(rng, x, x, 1e-4, 1e-8)
assert g.algorithms.optimize.lbfgs(maxiter=50, eps=1e-10)(q)(x, x)
assert x[0] is z and q(x) < 1e-16

# a run keeps its pairs: calls of 5 iterations follow one call of many
x = [complex(-1.2, 0.3), complex(1.0, -0.2)]
x1 = list(x)
g.algorithms.optimize.lbfgs(maxiter=40, eps=0.0)(rosenbrock())(x1, x1)
run = g.algorithms.optimize.lbfgs(maxiter=5, eps=0.0).on(x)
for i in range(8):
    run(rosenbrock())
assert x == x1
x = [complex(-1.2, 0.3), complex(1.0, -0.2)]
run = g.algorithms.optimize.lbfgs(maxiter=5, eps=1e-10).on(x)
for i in range(60):
    if run(rosenbrock()):
        break
g.message(f"lbfgs rosenbrock in calls of 5 iterations: {x} after {i + 1} calls")
assert abs(x[0] - 1) < 1e-8 and abs(x[1] - 1) < 1e-8


# failure_value: a RuntimeError at a trial point (here: outside |a| < 3) is a
# large value, the line search steps back
class guarded(rosenbrock):
    def __call__(self, fields):
        if abs(fields[0]) > 3:
            raise RuntimeError("outside")
        return super().__call__(fields)


x = [complex(-1.2, 0.3), complex(1.0, -0.2)]
assert g.algorithms.optimize.lbfgs(maxiter=200, eps=1e-10, failure_value=1e10)(guarded())(x, x)
assert abs(x[0] - 1) < 1e-8 and abs(x[1] - 1) < 1e-8

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
