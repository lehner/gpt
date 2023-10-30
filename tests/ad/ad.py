#!/usr/bin/env python3
#
# Authors: Christoph Lehner 2023
#
import gpt as g

rng = g.random("test")

for prec in [g.double]:
    grid = g.grid([4, 4, 4, 4], prec)
    g.message(f"Testing in precision {prec.__name__}")

    # create a simple two-layer network
    la1 = g.mspincolor(grid)
    lb1 = g.vspincolor(grid)
    la2 = g.mspincolor(grid)
    lb2 = g.vspincolor(grid)

    lin = g.vspincolor(grid)
    rng.cnormal([la1, la2, lb1, lb2, lin])

    # first test pure reverse ad (backpropagation)
    rad = g.ad.reverse

    # now create the compute graph for automatic differentiation
    a1 = rad.node(la1)
    a2 = rad.node(la2)
    b1 = rad.node(lb1)
    b2 = rad.node(lb2)
    # x = rad.node(lin, with_gradient=False)
    x = rad.node(lin)

    for c, learn_rate in [
        (rad.norm2(rad.relu(a2 * rad.relu(a1 * x + b1) + b2) - x), 1e-3),
        (rad.norm2(2.0 * a2 * a1 * rad.relu(a1 * x + b1) - 3.5 * a2 * b2), 1e-5),
    ]:
        v0 = c()

        # numerically test derivatives
        eps = prec.eps**0.5
        g.message(f"Numerical derivatives with eps = {eps}")
        for var in [a1, a2, b1, b2, x]:
            lt = g.lattice(var.value)
            rng.cnormal([lt])
            var.value += lt * eps
            v1 = c(with_gradients=False)
            var.value -= 2 * lt * eps
            v2 = c(with_gradients=False)
            var.value += lt * eps

            num_result = (v1 - v2).real / eps / 2.0
            ad_result = g.inner_product(var.gradient, lt).real
            err = abs(num_result / ad_result - 1)
            g.message(f"Error: {err}")
            assert err < 1e-4

        # create something to minimize
        class fnc(g.group.differentiable_functional):
            def __init__(self):
                pass

            def __call__(self, fields):
                global la1, lb1, la2, lb2
                la1 @= fields[0]
                lb1 @= fields[1]
                la2 @= fields[2]
                lb2 @= fields[3]
                return c(with_gradients=False).real

            def gradient(self, fields, dfields):
                c()
                assert dfields == fields
                return [a1.gradient, b1.gradient, a2.gradient, b2.gradient]

        f = fnc()
        ff = [la1, lb1, la2, lb2]
        v0 = f(ff)
        opt = g.algorithms.optimize.gradient_descent(maxiter=40, eps=1e-7, step=learn_rate)
        opt(f)(ff, ff)
        v1 = f(ff)
        g.message(f"Reduced value from {v0} to {v1}")
        assert v1 < v0
