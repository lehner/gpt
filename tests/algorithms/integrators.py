#!/usr/bin/env python3
#
# Authors: Christoph Lehner 2020
#
import gpt as g

# load configuration
rng = g.random("test")
grid = g.grid([4, 4, 4, 8], g.double)

U = g.mcolor(grid)
V = g.mcolor(grid)
rng.element([U, V])


def dC(u):
    r = g(g.qcd.gauge.project.traceless_anti_hermitian(g(u * V)) * (1j))
    r.otype = u.otype.cartesian()
    return r


# integrate using RK4
eps = 0.01
U_eps = g.algorithms.integrator.runge_kutta_4(U, dC, eps)

# integrate manually with lower-order routine and smaller step
t = 0.0
U_delta = g.copy(U)
while t < eps:
    delta = eps / 100.0
    U_delta = g(U_delta + g.matrix.exp(1j * dC(U_delta) * delta) * U_delta)
    t += delta

eps_test = (g.norm2(U_delta - U_eps) / g.norm2(U_eps)) ** 0.5
g.message(eps_test)

# TODO: need to complete this but booster is down now

