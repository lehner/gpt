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

    # relu without leakage
    relu = g.component.relu()

    # test a few simple models
    for c, learn_rate in [
        (g.norm2(a1) + 3.0 * g.norm2(a2 * b1 + b2 + t1 * x), 1e-1),
        (g.norm2(relu(a2 * relu(a1 * x + b1) + g.adj(t1 * x + b2)) - x), 1e-1),
        (
            g.norm2(
                2.0 * a2 * t1 * a1 * relu(a1 * x + b1)
                - 3.5 * a2 * b2 * g.trace(a1)
                - 1.5 * g.color_trace(a1) * a2 * b2
                + 2.5 * a2 * g.spin_trace(a1) * b2
                + t1 * g.cshift(a1 * x, 1, -1)
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
exa = g.inner_product(ly, lx * ly)[dm]
err = abs(est - exa) / abs(exa)
g.message(f"d <.,.> / dm : {err}")
assert err < 1e-7


est = (scale(eps) + scale(-eps) - 2 * scale(0)) / eps**2 / 2
exa = g.inner_product(ly, lx * ly)[dm**2]
err = abs(est - exa) / abs(exa)
g.message(f"d <.,.> / dm**2 : {err}")
assert err < 1e-5


test = g.norm2(g.cshift(g.cshift(lz, 0, 1), 0, -1) - lz)
err = abs(test[1] + test[dm] + test[alpha] + test[alpha**2] + test[dm**2] + test[dm * alpha])
assert err < 1e-6


#####################################
# combined forward/reverse AD tests
#####################################
dbeta = g.ad.forward.infinitesimal("dbeta")
On = g.ad.forward.landau(dbeta**2)

U = []
for mu in range(4):
    U_mu_0 = rng.element(g.mcolor(grid))
    U_mu = g.ad.forward.series(U_mu_0, On)
    U_mu[dbeta] = g(1j * U_mu_0 * rng.element(g.lattice(grid, U_mu_0.otype.cartesian())))
    U.append(U_mu)

Id = g.ad.forward.series(g.identity(U_mu_0), On)

# unitarity
for mu in range(4):
    eps2 = g.norm2(U[mu] * g.adj(U[mu]) - Id)
    eps2 = eps2[1].real + eps2[dbeta].real
    assert eps2 < 1e-25

# compare to wilson action
a = g.qcd.gauge.action.wilson(1)


def plaquette(U):
    res = None
    for mu in range(4):
        for nu in range(mu):
            nn = g(
                g.trace(
                    g.adj(U[nu]) * U[mu] * g.cshift(U[nu], mu, 1) * g.cshift(g.adj(U[mu]), nu, 1)
                )
            )
            if res is None:
                res = nn
            else:
                res += nn
    res = (res + g.adj(res)) / 12 / 3
    vol = U[0].grid.gsites
    Nd = 4
    res = g.sum(res) / vol
    return (Nd - 1) * Nd * vol / 2.0 * (1.0 - res)


# first test action
res = plaquette(U)
U1 = [u[1] for u in U]
err = abs(res[1] - a(U1))
g.message(f"Action test: {err}")
assert err < 1e-6

U_2 = [g.ad.reverse.node(u) for u in U]
res = plaquette(U_2)()
err = abs(res[1] - a(U1))
g.message(f"Action test (FW/REV): {err}")
assert err < 1e-6

# gradient test
gradients = a.gradient(U1, U1)
gradients2 = [u.gradient[1] for u in U_2]
for mu in range(4):
    err = (g.norm2(gradients[mu] - gradients2[mu]) / g.norm2(gradients[mu])) ** 0.5
    g.message(f"Gradient test [{mu}]: {err}")
    assert err < 1e-10

# numerical derivative of action value test
eps = 1e-4
U_plus = [g(u[1] + eps * u[dbeta]) for u in U]
U_minus = [g(u[1] - eps * u[dbeta]) for u in U]
da_dbeta = (plaquette(U_plus) - plaquette(U_minus)) / 2 / eps
err = abs(da_dbeta - res[dbeta]) / abs(da_dbeta)
g.message(f"Numerical action derivative test: {err}")
assert err < 1e-5

# numerical derivative of gradient test
for mu in range(4):
    dg_dbeta = g((a.gradient(U_plus, U_plus)[mu] - a.gradient(U_minus, U_minus)[mu]) / 2 / eps)
    err = (g.norm2(dg_dbeta - U_2[mu].gradient[dbeta]) / g.norm2(dg_dbeta)) ** 0.5
    g.message(f"Numerical action gradient [{mu}] derivative test: {err}")
    assert err < 1e-5
