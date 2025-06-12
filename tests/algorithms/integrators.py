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

    eps_test = g.norm2(U_delta - U_eps) ** 0.5 / U_eps.grid.gsites / U_eps.otype.nfloats / eps
    eps_ref = 10 * delta**2.0
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
eps_ref = eps**3.0
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
l = 0.05
a1 = g.qcd.scalar.action.phi4(k, l)

# starting config
q[:] = 0
rng.element(p)
p0 @= p

# evolution
tau = 1.0

# integrators
sympl = g.algorithms.integrator.symplectic
log = sympl.log()

ip = sympl.update_p(p, log(lambda: a1.gradient(q, q), "ip"), "P1")
iq = sympl.update_q(q, log(lambda: a0.gradient(p, p), "iq"), "Q1")
ip_fg = sympl.update_p_force_gradient(q, iq, p, ip, ip, "P_FG")

# ref solution obtained with Euler scheme
M = 1000
eps = tau / M
for k in range(M):
    ip(eps)
    iq(eps)
qref = g.lattice(q)
qref @= q

# ref solution for implicit scheme
ip_imp_euler = sympl.update_p(p, lambda: g(a1.gradient(q, q) + 0.01 * a1.gradient(p, p)))
iq_imp_euler = sympl.update_q(q, lambda: g(a0.gradient(p, p) + 0.01 * a0.gradient(q, q)))
q[:] = 0
p @= p0
eps = tau / M
for k in range(M):
    ip_imp_euler(eps / 2)
    iq_imp_euler(eps)
    ip_imp_euler(eps / 2)
qref_imp = g.lattice(q)
qref_imp @= q

# for test of multiple time-scale integrators
ip1 = sympl.update_p(p, log(lambda: g(0.8 * a1.gradient(q, q)), "ip"))
ip2 = sympl.update_p(p, log(lambda: g(0.2 * a1.gradient(q, q)), "ip"))

# for test of implicit integrators
p2 = g.copy(p)
q2 = g.copy(q)
_ip_imp = sympl.update_p(p, log(lambda: g(a1.gradient(q, q) + 0.01 * a1.gradient(p2, p2)), "ip"))
_ip1_imp = sympl.update_p(
    p, log(lambda: g(0.8 * a1.gradient(q, q) + 0.008 * a1.gradient(p2, p2)), "ip")
)
_ip2_imp = sympl.update_p(
    p, log(lambda: g(0.2 * a1.gradient(q, q) + 0.002 * a1.gradient(p2, p2)), "ip")
)
_iq_imp = sympl.update_q(q, log(lambda: g(a0.gradient(p, p) + 0.01 * a0.gradient(q2, q2)), "iq"))
ip_imp = sympl.implicit_update(p, p2, _ip_imp, eps=1e-16, tag="P")
ip1_imp = sympl.implicit_update(p, p2, _ip1_imp, eps=1e-16, tag="P")
ip2_imp = sympl.implicit_update(p, p2, _ip2_imp, eps=1e-16, tag="P")
iq_imp = sympl.implicit_update(q, q2, _iq_imp, eps=1e-16, tag="Q")
ip_fg_imp = sympl.implicit_update(
    [q, p],
    [q2, p2],
    sympl.update_p_force_gradient([q, q2], _iq_imp, [p, p2], _ip_imp, _ip_imp, "P_FG"),
    eps=1e-16,
    tag="FG_P",
)

nsteps = 20
integrator = [
    sympl.OMF2_force_gradient(nsteps, ip_imp, iq_imp, ip_fg_imp),
    sympl.leap_frog(nsteps, ip, iq),
    sympl.OMF2(nsteps, ip, iq),
    sympl.OMF2_force_gradient(nsteps, ip, iq, ip_fg),
    sympl.OMF4(nsteps, ip, iq),
    sympl.OMF2(12, ip2, sympl.OMF4(1, ip1, iq)),
    sympl.OMF2(12, ip2, sympl.OMF4(2, ip1, iq)),
    sympl.OMF2(nsteps, ip_imp, iq_imp),
    sympl.OMF2(12, ip2_imp, sympl.OMF4(1, ip1_imp, iq_imp)),
]
criterion = [1e-8, 1e-5, 1e-7, 1e-11, 1e-11, 1e-8, 1e-8, 1e-7, 1e-8]
refs = [qref_imp, qref, qref, qref, qref, qref, qref, qref_imp, qref_imp]

for i in range(len(integrator)):
    # initial config
    q[:] = 0
    p @= p0

    # print/log
    log.reset()
    sep = "--------------------------------------------------------------------------------"
    g.message(f"{sep}\n{integrator[i]}\n{sep}\n")

    # integrate
    integrator[i](tau)

    eps = g.norm2(q - refs[i])
    g.message(f"{integrator[i].__name__ : <10}: |q - qref|^2 = {eps:.4e}")
    assert eps < criterion[i]

    # test reversibility
    integrator[i](-tau)
    eps = g.norm2(q)
    g.message(f"{integrator[i].__name__ : <10} reversibility test: {eps:.4e}")
    assert eps < 1e-26

    g.message("Max force = ", max(log.get("ip")))
    g.message(f"Timing:\n{log.time}")

