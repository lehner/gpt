#!/usr/bin/env python3
#
# g.algorithms.nonlinear: fixed_point (iteration control, failure, Newton acceleration)
#
import gpt as g
import numpy as np

rng = g.random("test")
grid = g.grid([4, 4, 4, 8], g.double)
b = g.complex(grid)
rng.cnormal(b)

# a contraction x <- x / 2 + b (fixed point 2 b), rate 1/2
fp = g.algorithms.nonlinear.fixed_point(eps=1e-13, maxiter=100)
x = g.complex(grid)
x[:] = 0


def step(x):
    x[0] @= 0.5 * x[0] + b


changes = []
fp([x], step, monitor=lambda i, c: changes.append(c))
eps = g.norm2(x - 2 * b) / g.norm2(b)
g.message(
    f"contraction: {len(fp.history)} iterations, rate {fp.rate(4)}, |x - 2 b|^2 / |b|^2 = {eps}"
)
assert fp.converged and eps < 1e-24 and abs(fp.rate(4) - 0.5) < 1e-3
assert len(changes) == len(fp.history) and abs(changes[-1] ** 2 - fp.history[-1]) < 1e-30

# not a contraction: x <- 2 x + b
x[:] = 0
try:
    fp([x], lambda x: x[0].__imatmul__(g(2.0 * x[0] + b)))
    assert False
except RuntimeError as e:
    g.message(f"not a contraction: {e}")
fp2 = g.algorithms.nonlinear.fixed_point(eps=1e-13, maxiter=10, raise_on_failure=False)
x[:] = 0
fp2([x], lambda x: x[0].__imatmul__(g(2.0 * x[0] + b)))
assert not fp2.converged and len(fp2.history) == 10

# a slow iteration for F(x) = x^3 + x - c = 0 (real fields), x <- x - F(x) / 20,
# accelerated by Newton steps d = -F(x) / (3 x^2 + 1)
c = g.complex(grid)
rng.uniform_real(c, min=-1, max=1)
one = g.complex(grid)
one[:] = 1


def F(x):
    return g(x * x * x + x - c)


def slow(x):
    x[0] @= x[0] - 0.05 * F(x[0])


newton = g.algorithms.nonlinear.fixed_point.newton(
    lambda x: [F(x[0])],
    lambda x, r: [g(-1.0 * r[0] * g.component.inv(g(3.0 * x[0] * x[0] + one)))],
)
for accelerated in [False, True]:
    x[:] = 0
    fp = g.algorithms.nonlinear.fixed_point(eps=1e-13, maxiter=1000)
    fp([x], slow, newton if accelerated else None)
    residual = g.norm2(F(x)) / grid.gsites
    g.message(
        f"cubic, accelerated={accelerated}: {len(fp.history)} iterations ({fp.accelerated_iterations} Newton), |F(x)|^2 per site {residual:.3e}"
    )
    assert fp.converged and residual < 1e-22
    assert (fp.accelerated_iterations > 0) == accelerated
    if accelerated:
        assert len(fp.history) < 20
