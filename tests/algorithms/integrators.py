#!/usr/bin/env python3
#
# Authors: Christoph Lehner 2020
#
import gpt as g

# load configuration
rng = g.random("test")
grid = g.grid([4, 4, 4, 8], g.double)

def dC_su_n(u):
    r = g(g.qcd.gauge.project.traceless_anti_hermitian(g(u * V * u * u * V)) * (1j))
    r.otype = u.otype.cartesian()
    return r

def dC_u1(u):
    r = g(g.component.log(u) / 1j)
    r @= r*r - r*r*r
    r.otype = u.otype.cartesian()
    return r

for group, dC in [(g.mcolor, dC_su_n), (g.u1,dC_u1)]:
    U = group(grid)
    V = group(grid)
    rng.element([U, V])

    # integrate using RK4
    eps = 0.1
    U_eps = g.algorithms.integrator.runge_kutta_4(U, dC, eps)

    # integrate manually with lower-order routine and smaller step
    t = 0.0
    U_delta = g.copy(U)
    N_steps = 100
    delta = eps / N_steps
    for i in range(N_steps):
        U_delta @= g.matrix.exp(1j * dC(U_delta) * delta) * U_delta
        
    eps_test = g.norm2(U_delta - U_eps)**0.5 / U_eps.grid.gsites / U_eps.otype.nfloats / eps
    eps_ref = 10*delta**2.
    g.message(f"Test on {U.otype.__name__}: {eps_test} < {eps_ref}")
    assert eps_test < eps_ref

# finally integrate a simple non-linear DGL
# y'(t) = y(t)**2
# y(0)  = 1
# expected: y(t) = 1.0 / (1.0 - t)
U = g.complex(grid)
U[:] = 1.0
eps = 0.01
U_eps = g.algorithms.integrator.runge_kutta_4(U, lambda u: g(u*u), eps)[0,0,0,0]
U_exp = 1.0 / (1.0 - eps)
eps_test = abs(U_eps - U_exp) / eps
eps_ref = eps ** 3.0
g.message(f"Test on geometric series: {eps_test} < {eps_ref}")
assert eps_test < eps_ref
