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

    # randomize values
    rng.cnormal([a1.value, a2.value, b1.value, b2.value, x.value, t1.value])

    # test a few simple models
    for c, learn_rate in [
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
        v0 = c()

        # numerically test derivatives
        eps = prec.eps**0.5
        g.message(f"Numerical derivatives with eps = {eps}")
        for var in [a1, a2, b1, b2, x, t1]:
            if isinstance(var.value, g.lattice):
                lt = g.lattice(var.value)
            else:
                lt = var.value.copy()
            rng.cnormal(lt)
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
        class fnc(g.group.differentiable_functional):
            def __init__(self):
                pass

            def __call__(self, fields):
                global a1, b1, a2, b2, t1
                a1.value @= fields[0]
                b1.value @= fields[1]
                a2.value @= fields[2]
                b2.value @= fields[3]
                t1.value @= fields[4]
                return c(with_gradients=False).real

            def gradient(self, fields, dfields):
                c()
                assert dfields == fields
                return [a1.gradient, b1.gradient, a2.gradient, b2.gradient, t1.gradient]

        f = fnc()
        ff = [a1.value, b1.value, a2.value, b2.value, t1.value]
        v0 = f(ff)
        opt = g.algorithms.optimize.adam(maxiter=40, eps=1e-7, alpha=learn_rate)
        opt(f)(ff, ff)
        v1 = f(ff)
        g.message(f"Reduced value from {v0} to {v1} with Adam")
        assert v1 < v0
