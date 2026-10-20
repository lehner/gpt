#!/usr/bin/env python3
#
# Authors: Christoph Lehner 2026
#
# Differentiable symplectic integrators (g.ad.reverse.symplectic) and
# g.group.compose of nodes
#
import gpt as g

rad = g.ad.reverse
sympl = g.algorithms.integrator.symplectic
dsympl = g.ad.reverse.symplectic
rng = g.random("symplectic")


def rel(x, y):
    x, y = g.util.to_list(x), g.util.to_list(y)
    return (sum(g.norm2(a - b) for a, b in zip(x, y)) / sum(g.norm2(b) for b in y)) ** 0.5


def values(xs):
    return [x.value for x in xs]


def check(name, eps, tol):
    g.message(f"{name}: {eps:.3e} (< {tol:.0e})")
    assert eps < tol


class phi4(g.qcd.scalar.action.phi4):
    # the value in node operations (the production value is not node-capable)
    def __call__(self, phi):
        p = g.util.to_list(phi)[0]
        J = g.cshift(p, 0, 1)
        for mu in range(1, p.grid.nd):
            J = J + g.cshift(p, mu, 1)
        p2 = g.sum(p * p)
        act = -2.0 * self.kappa * g.sum(J * p) + p2
        act = act + self.l * (g.sum(p * p * p * p) - 2.0 * p2 + float(p.grid.fsites))
        return g.component.real(act)


class scaled(g.core.group.differentiable_functional):
    # Re(c) S(U) with c the last field (a real parameter of the action:
    # numbers are complex)
    def __init__(self, S):
        self.S = S
        self.__name__ = f"scaled({S.__name__})"

    def __call__(self, fields):
        return g.component.real(fields[-1]) * self.S(fields[:-1])

    def gradient(self, fields, dfields):
        U, c = fields[:-1], fields[-1]
        return [complex(self.S(U)) if d is c else g(c.real * self.S.gradient(U, [d])[0]) for d in dfields]


grid = g.grid([4, 4, 4, 8], g.double)
a0 = g.qcd.scalar.action.mass_term()
w1 = g.qcd.gauge.action.wilson(1.0)

#
# g.group.compose of nodes: SU(3) links and algebra elements, values and gradients
#
U0 = g.qcd.gauge.random(grid, rng, scale=0.3)
A0 = rng.normal_element(g.group.cartesian(U0))
x = [rad.node(g.copy(u)) for u in U0] + [rad.node(g.copy(a)) for a in A0]
y = g.group.compose([g(0.3 * a) for a in x[4:8]], x[0:4])
check("compose of nodes vs plain (links)", rel([n(with_gradients=False) for n in y], g.group.compose([g(0.3 * a) for a in A0], U0)), 1e-14)
assert all(n.otype.__name__ == U0[0].otype.__name__ for n in y)
z = g.group.compose(x[4], g(x[4] * 0.5))
check("compose of nodes vs plain (algebra)", rel(z(with_gradients=False), g(1.5 * A0[0])), 1e-15)
E = w1(y) + g.component.real(g.sum(g.trace(z * z)))
E.functional(*x).assert_gradient_error(rng, values(x), values(x), 1e-4, 1e-8)

#
# 1) real scalar field (phi^4), fields [q, p]
#
q0 = g.real(grid)
rng.normal(q0)
q0 @= 0.5 * q0
p0 = g.group.cartesian(q0)
rng.element(p0)
a1 = phi4(0.1234, 0.05)

# forces: the recorded gradient against the plain one
for S, x in [(a1, q0), (a0, p0)]:
    plain = dsympl.gradient(S)([x])[0]
    n = rad.node(g.copy(x), infinitesimal_to_cartesian=False)
    frc = dsympl.gradient(S).recorded([n])[0]
    check(f"recorded force of {S.__name__} vs plain", rel(frc(with_gradients=False), plain), 1e-14)

ip_d = dsympl.update_p([1], dsympl.gradient(a1), [0])
iq_d = dsympl.update_q([0], dsympl.gradient(a0), [1])
ip1_d = dsympl.update_p([1], lambda f: [g(0.8 * x) for x in dsympl.gradient(a1)(f)], [0])
ip2_d = dsympl.update_p([1], lambda f: [g(0.2 * x) for x in dsympl.gradient(a1)(f)], [0])

# the in-place reference
q, p = g.copy(q0), g.copy(p0)
ip = sympl.update_p(p, lambda: a1.gradient(q, q))
iq = sympl.update_q(q, lambda: a0.gradient(p, p))
ip1 = sympl.update_p(p, lambda: g(0.8 * a1.gradient(q, q)))
ip2 = sympl.update_p(p, lambda: g(0.2 * a1.gradient(q, q)))

tau = 0.8
c_q = g.real(grid)
rng.normal(c_q)
c_p = g.lattice(p0)
rng.element(c_p)


def observable(x):
    # a generic scalar of the final fields: <c_q, q^3> + <c_p, p> + |p q|^2
    q, p = x
    return g.component.real(g.sum(c_q * q * q * q) + g.sum(c_p * p) + g.sum(p * q * p * q))


for name, ref, md in [
    ("leap_frog", sympl.leap_frog(6, ip, iq), dsympl.leap_frog(6, ip_d, iq_d)),
    ("OMF2", sympl.OMF2(4, ip, iq), dsympl.OMF2(4, ip_d, iq_d)),
    ("OMF2(OMF4)", sympl.OMF2(3, ip2, sympl.OMF4(1, ip1, iq)), dsympl.OMF2(3, ip2_d, dsympl.OMF4(1, ip1_d, iq_d))),
]:
    g.message(f"---- phi^4 {name}\n{md}")
    # plain: bit-identical to the in-place integrator
    g.copy(q, q0)
    g.copy(p, p0)
    ref(tau)
    y = md([q0, p0], tau)
    check(f"{name}: plain vs in-place", rel(y, [q, p]), 1e-300)

    # nodes: the same values, gradients against finite differences of E
    x = [rad.node(g.copy(q0)), rad.node(g.copy(p0))]
    yn = md(x, tau)
    check(f"{name}: node forward vs plain", rel([n(with_gradients=False) for n in yn], y), 1e-14)
    observable(yn).functional(*x).assert_gradient_error(rng, values(x), values(x), 1e-4, 1e-7)

    # reversibility of the node trajectory (momentum flip); fresh leaves (the
    # functional left the old ones at finite-difference points)
    x = [rad.node(g.copy(q0)), rad.node(g.copy(p0))]
    yn = md(x, tau)
    w = [n(with_gradients=False) for n in md([yn[0], -1.0 * yn[1]], tau)]
    check(f"{name}: reversibility", rel([w[0], g(-w[1])], [q0, p0]), 1e-13)

# second order: Hessian-vector product against a finite difference of the gradient
md = dsympl.leap_frog(3, ip_d, iq_d)
dq = g.real(grid)
rng.normal(dq)
dp = g.lattice(p0)
rng.element(dp)


def gradient_at(qv, pv):
    x = [rad.node(g.copy(qv)), rad.node(g.copy(pv))]
    observable(md(x, tau)).backward()
    return [g.copy(n.gradient) for n in x]


x = [rad.node(g.copy(q0)), rad.node(g.copy(p0))]
observable(md(x, tau)).backward(create_graph=True)
gq, gp = x[0].gradient, x[1].gradient
x[0].flow = x[1].flow = None
g.component.real(g.sum(dq * gq) + g.sum(dp * gp)).backward()
hvp = [g.copy(n.gradient) for n in x]
h = 1e-4
gplus = gradient_at(g(q0 + h * dq), g(p0 + h * dp))
gminus = gradient_at(g(q0 - h * dq), g(p0 - h * dp))
fd = [g((a - b) / (2 * h)) for a, b in zip(gplus, gminus)]
check("leap_frog: Hessian-vector product vs finite difference", rel(hvp, fd), 1e-6)

#
# 2) SU(3) Wilson action with beta as a field: [U_0..U_3, P_0..P_3, beta]
#
P0 = g.group.cartesian(U0)
a0.draw(P0, rng)
beta = 5.5
Sg = scaled(w1)
iU, iP = list(range(4)), list(range(4, 8))
iq_d = dsympl.update_q(iU, lambda f: f, iP)

U, P = g.copy(U0), g.copy(P0)
ip = sympl.update_p(P, lambda: Sg.gradient(U + [beta], U))
iq = sympl.update_q(U, lambda: a0.gradient(P, P))

tau = 0.4
c = rng.normal_element(g.group.cartesian(U0))


def gauge_observable(y):
    # plaquette action of the final links + a linear term in the momenta
    return w1(y[0:4]) + sum(g.component.real(g.sum(g.trace(ci * pi))) for ci, pi in zip(c, y[4:8]))


def leaves(fields):
    return [rad.node(g.copy(f)) if isinstance(f, g.lattice) else rad.node(f) for f in fields]


forces = [("recorded", dsympl.gradient(Sg, 4)), ("finite difference", dsympl.gradient_finite_difference(Sg, 4))]
gradients = {}
for force_name, force in forces:
    ip_d = dsympl.update_p(iP, force, iU + [8])
    for name, ref, md in [
        ("leap_frog", sympl.leap_frog(3, ip, iq), dsympl.leap_frog(3, ip_d, iq_d)),
        ("OMF2", sympl.OMF2(2, ip, iq), dsympl.OMF2(2, ip_d, iq_d)),
    ]:
        tag = f"SU(3) {name}, {force_name} force"
        g.message(f"---- {tag}")
        g.copy(U, U0)
        g.copy(P, P0)
        ref(tau)
        y = md(U0 + P0 + [beta], tau)
        check(f"{tag}: plain vs in-place", rel(y[0:8], U + P), 1e-300)

        x = leaves(U0 + P0 + [beta])
        yn = md(x, tau)
        check(f"{tag}: node forward vs plain", rel([n(with_gradients=False) for n in yn[0:8]], y[0:8]), 1e-13)
        gauge_observable(yn).functional(*x).assert_gradient_error(rng, values(x), values(x), 1e-4, 1e-7)

        # dE/dbeta against a central difference
        x = leaves(U0 + P0 + [beta])
        gauge_observable(md(x, tau)).backward()
        gradients[force_name, name] = [n.gradient for n in x]
        dbeta = x[8].gradient.real
        h = 1e-4
        fd = (gauge_observable(md(U0 + P0 + [beta + h], tau)) - gauge_observable(md(U0 + P0 + [beta - h], tau))) / (2 * h)
        check(f"{tag}: dE/dbeta = {dbeta} vs finite difference {fd}", abs(dbeta - fd) / abs(fd), 1e-7)

for name in ["leap_frog", "OMF2"]:
    r, f = gradients["recorded", name], gradients["finite difference", name]
    check(f"SU(3) {name}: finite difference vs recorded force, gradient w.r.t. U, P", rel(f[0:8], r[0:8]), 1e-10)
    check(f"SU(3) {name}: finite difference vs recorded force, dE/dbeta", abs(f[8] - r[8]) / abs(r[8]), 1e-9)

# second order on the gauge field: d^2/dh^2 E(md(x(h))) along x(h) = (exp(h a) U,
# P + h b) from the recorded gradient (contracted twice) and from a finite
# difference.  beta is fixed here: in a recorded pass the vjp of the force
# checkpoint is a checkpoint whose outputs are the flows into its inputs, and
# checkpoint outputs must share one container (links and a number do not)
md = dsympl.OMF2(2, dsympl.update_p(iP, dsympl.gradient(g.qcd.gauge.action.wilson(beta)), iU), iq_d)
a = rng.normal_element(g.group.cartesian(U0))
b = rng.normal_element(g.group.cartesian(U0))
d = a + b


def E_at(h):
    U = [g(u) for u in g.group.compose([g(h * ai) for ai in a], U0)]
    P = [g(p + h * bi) for bi, p in zip(b, P0)]
    return gauge_observable(md(U + P, tau))


x = leaves(U0 + P0)
gauge_observable(md(x, tau)).backward(create_graph=True)
gr = [n.gradient for n in x]
for n in x:
    n.flow = None
g.group.inner_product(d, gr).backward()
d2_ad = g.group.inner_product(d, [n.gradient for n in x])
h = 3e-4
d2_fd = (E_at(h) - 2 * E_at(0.0) + E_at(-h)) / h**2
check(f"SU(3) OMF2: second derivative {d2_ad} vs finite difference {d2_fd}", abs(d2_ad - d2_fd) / abs(d2_fd), 1e-6)

#
# 3) an action without node-capable value: two transport layers (rho a field
#    of the transformations) on the Wilson action with their log dets,
#    fields [U_0..U_3, P_0..P_3, rho]
#
mu = 0
even, odd = g.even_odd_projectors(grid)
full = g(even + odd)
rho = g.complex(grid)
rho[:] = 0.1
description = [(rho, g.path().f(nu).f(mu).b(nu).b(mu)) for nu in range(4) if nu != mu] + [
    (rho, g.path().b(nu).f(mu).f(nu).b(mu)) for nu in range(4) if nu != mu
]


def layer(P1):
    return g.qcd.gauge.smear.directional_parallel_transport(U0, description, mu, full, P1, [rho])


phi_1, phi_2 = layer(odd), layer(even)
indices = list(range(5))
S = g.qcd.gauge.action.wilson(5.5)
S = S.transformed(phi_2, indices=indices, projection=[0, 1, 2, 3])
S = S.transformed(phi_1, indices=indices, projection=indices)
ld_2 = phi_2.action_log_det_jacobian().transformed(phi_1, indices=indices, projection=indices)
S = S + phi_1.action_log_det_jacobian() + ld_2

ip_d = dsympl.update_p(iP, dsympl.gradient_finite_difference(S, 4), iU + [8])
for name, md in [("leap_frog", dsympl.leap_frog(2, ip_d, iq_d)), ("OMF2", dsympl.OMF2(2, ip_d, iq_d))]:
    tag = f"transported Wilson action + log dets, {name}"
    g.message(f"---- {tag}")
    x = leaves(U0 + P0 + [rho])
    gauge_observable(md(x, tau)).functional(*x).assert_gradient_error(rng, values(x), values(x), 1e-4, 1e-7)

    # dE/drho for a global shift of rho against a 4-point difference (a
    # central difference has an h^2 error of 1e-6 here)
    x = leaves(U0 + P0 + [rho])
    gauge_observable(md(x, tau)).backward()
    drho = g.sum(x[8].gradient).real
    h = 1e-4
    E = {s: gauge_observable(md(U0 + P0 + [g(rho + s * h * full)], tau)) for s in [-2, -1, 1, 2]}
    fd = (8 * (E[1] - E[-1]) - (E[2] - E[-2])) / (12 * h)
    check(f"{tag}: dE/drho = {drho} vs finite difference {fd}", abs(drho - fd) / abs(fd), 1e-8)
