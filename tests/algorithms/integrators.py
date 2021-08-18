#!/usr/bin/env python3
#
# Authors: Christoph Lehner, Mattia Bruno 2021
#
import gpt as g

# load configuration
rng = g.random("test")
grid = g.grid([4, 4, 4, 8], g.double)

#
# Test integrationg ODEs
#
def dC_su_n(u):
    r = g(g.qcd.gauge.project.traceless_anti_hermitian(g(u * V * u * u * V)) * (1j))
    r.otype = u.otype.cartesian()
    return r


def dC_u1(u):
    r = g(g.component.log(u) / 1j)
    r @= r * r - r * r * r
    r.otype = u.otype.cartesian()
    return r


for group, dC in [(g.mcolor, dC_su_n), (g.u1, dC_u1)]:
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

    eps_test = (
        g.norm2(U_delta - U_eps) ** 0.5 / U_eps.grid.gsites / U_eps.otype.nfloats / eps
    )
    eps_ref = 10 * delta ** 2.0
    g.message(f"Test on {U.otype.__name__}: {eps_test} < {eps_ref}")
    assert eps_test < eps_ref

# finally integrate a simple non-linear ODE
# y'(t) = y(t)**2
# y(0)  = 1
# expected: y(t) = 1.0 / (1.0 - t)
U = g.complex(grid)
U[:] = 1.0
eps = 0.01
U_eps = g.algorithms.integrator.runge_kutta_4(U, lambda u: g(u * u), eps)[0, 0, 0, 0]
U_exp = 1.0 / (1.0 - eps)
eps_test = abs(U_eps - U_exp) / eps
eps_ref = eps ** 3.0
g.message(f"Test on geometric series: {eps_test} < {eps_ref}")
assert eps_test < eps_ref


#
# Test symplectic integrators below
#
# Use harmonic oscillator Hamiltonian
#
q = g.real(grid)
p = g.group.cartesian(q)
p0 = g.lattice(p)

# p^2 / 2m , m=1
a0 = g.qcd.scalar.action.mass_term()

# q^2 * k / 2
k = 0.1234
a1 = g.qcd.scalar.action.mass_term(k)

# starting config
q[:] = 0
rng.element(p)
p0 @= p

# evolution
tau = 1.0

# integrators
sympl = g.algorithms.integrator.symplectic

ip = sympl.update_p(p, lambda: a1.gradient(q, q))
iq = sympl.update_q(q, lambda: a0.gradient(p, p))

# ref solution obtained with Euler scheme
M = 1000
eps = tau / M
for k in range(M):
    ip(eps)
    iq(eps)
qref = g.lattice(q)
qref @= q

integrator = [sympl.leap_frog, sympl.OMF2, sympl.OMF4]
criterion = [1e-5, 1e-8, 1e-12]

for i in range(3):
    # initial config
    q[:] = 0
    p @= p0

    # solve
    integrator[i](10, ip, iq)(tau)

    eps = g.norm2(q - qref)
    print(f"{integrator[i].__name__ : <10}: |q - qref|^2 = {eps:.4e}")
    assert eps < criterion[i]
