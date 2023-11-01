#!/usr/bin/env python3
#
# Authors: Christoph Lehner 2023
#
import gpt as g

rng = g.random("test")

for prec in [g.double]:
    grid = g.grid([4, 4, 4, 4], prec)
    g.message(f"Testing in precision {prec.__name__}")

    # first test pure reverse ad (backpropagation)
    rad = g.ad.reverse

    # create the compute graph for automatic differentiation
    a1 = rad.node(g.mspincolor(grid))
    a2 = rad.node(g.mspincolor(grid))
    b1 = rad.node(g.vspincolor(grid))
    b2 = rad.node(g.vspincolor(grid))
    x = rad.node(g.vspincolor(grid))
    t1 = rad.node(g.tensor(a1.value.otype))

    # test a few simple models
    for c, learn_rate in [
        (rad.norm2(a1) + 3.0*rad.norm2(a2*b1 + b2 + t1*x), 1e-1),
        (rad.norm2(rad.relu(a2 * rad.relu(a1 * x + b1) + t1 * x + b2) - x), 1e-1),
        (
            rad.norm2(
                2.0 * a2 * t1 * a1 * rad.relu(a1 * x + b1)
                - 3.5 * a2 * b2
                + t1 * rad.cshift(a1 * x, 1, -1)
            ),
            1e-1,
        ),
    ]:
        # randomize values
        rng.cnormal([a1.value, a2.value, b1.value, b2.value, x.value, t1.value])

        v0 = c()

        # numerically test derivatives
        eps = prec.eps**0.5 * 100
        g.message(f"Numerical derivatives with eps = {eps}")
        for var in [a1, a2, b1, b2, x, t1]:
            lt = rng.cnormal(var.value.new())
            var.value += lt * eps
            v1 = c(with_gradients=False)
            var.value -= 2 * lt * eps
            v2 = c(with_gradients=False)
            var.value += lt * eps

            num_result = (v1 - v2).real / eps / 2.0 + 1e-15
            ad_result = g.inner_product(var.gradient, lt).real + 1e-15
            err = abs(num_result / ad_result - 1)
            g.message(f"Error: {err}")
            assert err < 1e-4

        # create something to minimize
        f = c.functional(a1, b1, a2, b2, t1)
        ff = [a1.value, b1.value, a2.value, b2.value, t1.value]
        v0 = f(ff)
        opt = g.algorithms.optimize.adam(maxiter=40, eps=1e-7, alpha=learn_rate)
        opt(f)(ff, ff)
        v1 = f(ff)
        g.message(f"Reduced value from {v0} to {v1} with Adam")
        assert v1 < v0
