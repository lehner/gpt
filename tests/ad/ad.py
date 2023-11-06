#!/usr/bin/env python3
#
# Authors: Christoph Lehner 2023
#
import gpt as g
import numpy as np

rng = g.random("test")

#####################################
# reverse AD tests
#####################################
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
        (rad.norm2(a1) + 3.0 * rad.norm2(a2 * b1 + b2 + t1 * x), 1e-1),
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


#####################################
# forward AD tests
#####################################
fad = g.ad.forward
dm = fad.infinitesimal("dm")
alpha = fad.infinitesimal("alpha")

assert (dm**4 * alpha).behaves_as(dm**3 * alpha)
assert fad.landau(dm**4, dm * alpha) + fad.landau(
    dm**2, alpha**2, dm**3 * alpha
) == fad.landau(dm**2, alpha**2, dm * alpha)

# landau O notation to keep series with O(1), O(dm), O(alpha), O(dm**2), O(alpha**2), O(alpha*dm) terms
# On determines terms that are neglected
On = fad.landau(dm**3, alpha**3, dm**2 * alpha, dm * alpha**2)
x = fad.series(3, On)
x[dm] = 2.2
x[alpha * dm] = 3.1612
x[alpha] = 4.88

y = x * x
assert abs(y[1] - 9) < 1e-8
assert abs(y[dm] - 13.2) < 1e-8
assert abs(y[alpha * dm] - 2 * 10.736 - 2 * 3 * 3.1612) < 1e-8
assert abs(y[alpha] - 29.28) < 1e-8


# define function
def fcos(x, nderiv):
    if nderiv % 2 == 0:
        return (-1) ** (nderiv // 2) * np.cos(x)
    else:
        return (-1) ** ((nderiv + 1) // 2) * np.sin(x)


fy = y.function(fcos)

eps = 1e-5

err = abs(np.cos(y[1]) - fy[1])
g.message(f"Error O(1): {err}")
assert err < 1e-8

err = abs((np.cos(y[1] + y[dm] * eps) - np.cos(y[1] - y[dm] * eps)) / eps / 2 - fy[dm])
g.message(f"Error O(dm): {err}")
assert err < 1e-5

err = abs((np.cos(y[1] + y[alpha] * eps) - np.cos(y[1] - y[alpha] * eps)) / eps / 2 - fy[alpha])
g.message(f"Error O(alpha): {err}")
assert err < 1e-5

err = abs(
    (
        np.cos(y[1] + y[dm] * eps + y[dm**2] * eps**2)
        + np.cos(y[1] - y[dm] * eps + y[dm**2] * eps**2)
        - 2 * np.cos(y[1])
    )
    / eps**2
    / 2
    - fy[dm**2]
)
g.message(f"Error O(dm**2): {err}")
assert err < 1e-5

err = abs(
    (
        np.cos(y[1] + y[alpha] * eps + y[alpha**2] * eps**2)
        + np.cos(y[1] - y[alpha] * eps + y[alpha**2] * eps**2)
        - 2 * np.cos(y[1])
    )
    / eps**2
    / 2
    - fy[alpha**2]
)
g.message(f"Error O(alpha**2): {err}")
assert err < 1e-5

err = abs(
    (
        +np.cos(y[1] + y[dm] * eps + y[alpha] * eps + y[alpha * dm] * eps**2)
        - np.cos(y[1] - y[dm] * eps + y[alpha] * eps - y[alpha * dm] * eps**2)
        - np.cos(y[1] + y[dm] * eps - y[alpha] * eps - y[alpha * dm] * eps**2)
        + np.cos(y[1] - y[dm] * eps - y[alpha] * eps + y[alpha * dm] * eps**2)
    )
    / eps**2
    / 4
    - fy[alpha * dm]
)
g.message(f"Error O(alpha*dm): {err}")
assert err < 1e-5

# now test with lattice
lx = fad.series(rng.cnormal(g.mcolor(grid)), On)
lx[dm] = rng.cnormal(g.mcolor(grid))
lx[alpha] = rng.cnormal(g.mcolor(grid))
ly = 2 * lx + 3 * lx * lx

ly = fad.series(rng.cnormal(g.vcolor(grid)), On)
ly[dm] = rng.cnormal(g.vcolor(grid))
ly[alpha] = rng.cnormal(g.vcolor(grid))

lz = lx * ly

eps = 1e-4


def scale(lam):
    return g.inner_product(
        g(ly[1] + ly[dm] * lam), g((lx[1] + lx[dm] * lam) * (ly[1] + ly[dm] * lam))
    )


est = (scale(eps) - scale(-eps)) / 2 / eps
exa = fad.inner_product(ly, lx * ly)[dm]
err = abs(est - exa) / abs(exa)
g.message(f"d <.,.> / dm : {err}")
assert err < 1e-7


est = (scale(eps) + scale(-eps) - 2 * scale(0)) / eps**2 / 2
exa = fad.inner_product(ly, lx * ly)[dm**2]
err = abs(est - exa) / abs(exa)
g.message(f"d <.,.> / dm**2 : {err}")
assert err < 1e-5


test = fad.norm2(fad.cshift(fad.cshift(lz, 0, 1), 0, -1) - lz)
g.message(test)

# TODO:
# - fad.series, rad.node need to play nice with g.eval
#   (inherit from g.evaluable)
# - fad.series, rad.node play nice with regular g.inner_product etc.
#   for use in regular algorithms; inherit from lattice_like which
#   should add maps to rad.inner_product, etc.
