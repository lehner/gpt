#!/usr/bin/env python3
#
# Training a gauge-covariant function f(P) of a site-local matrix field P,
# P(x) -> V(x) P(x) V(x)^dag, with f(V P V^dag) = V f(P) V^dag (stand-alone,
# reverse AD only).  P is the weighted plaquette loop sum through a link, the
# argument of loop_function in directional_parallel_transport.
#
import gpt as g
import numpy as np

rad = g.ad.reverse
rng = g.random("test")
grid = g.grid([4, 4, 4, 4], g.double)


class covariant_matrix_network:
    # C channels X_c of N x N matrices (all transforming like P), L residual
    # layers
    #   Y_c = sum_d (a_cd X_d + b_cd X_d^dag) + beta_c 1
    #   X_c <- X_c + Y_c Y_{c+1 mod C}
    # and the output f = P + sum_c w_c X_c, starting from X_c = P.  Weights are
    # complex scalars; every operation is covariant (scalar coefficients,
    # products, adjoints, the identity).  The same code runs on plain fields
    # (weights as numbers) and on nodes (weights as node leaves).
    def __init__(self, n_channels, n_layers):
        self.C = n_channels
        self.L = n_layers

    def weights(self, rng, scale=0.1):
        C = self.C

        def r():
            return complex(scale * (rng.normal() + 1j * rng.normal()) / np.sqrt(2))

        w = []
        for _ in range(self.L):
            w += [r() for _ in range(2 * C * C + C)]
        # a small output: f starts close to the identity map (with zero output
        # weights the inner weights would get no gradient)
        w += [0.1 * r() for _ in range(C)]
        return w

    def __call__(self, P, w, identity):
        C = self.C
        X = [P] * C
        k = 0
        for _ in range(self.L):
            Xa = [g.adj(x) for x in X]
            Y = []
            for c in range(C):
                y = w[k + 2 * C * C + c] * identity
                for d in range(C):
                    y = y + w[k + c * C + d] * X[d] + w[k + C * C + c * C + d] * Xa[d]
                Y.append(y)
            X = [g(X[c] + Y[c] * Y[(c + 1) % C]) for c in range(C)]
            k += 2 * C * C + C
        f = P
        for c in range(C):
            f = f + w[k + c] * X[c]
        return g(f)


def loop_sum(U, mu, rho):
    # rho * sum_nu (plaquettes at x through the link (x, mu), ending with b(mu))
    staple = None
    for nu in range(len(U)):
        if nu == mu:
            continue
        U_mu_nu = g.cshift(U[mu], nu, 1)
        up = U[nu] * U_mu_nu * g.adj(g.cshift(U[nu], mu, 1))
        U_nu_m = g.cshift(U[nu], nu, -1)
        down = g.adj(U_nu_m) * g.cshift(U[mu], nu, -1) * g.cshift(U_nu_m, mu, 1)
        s = g(up + down)
        staple = s if staple is None else g(staple + s)
    return g(rho * staple * g.adj(U[mu]))


def teacher(P):
    return g(P + 0.3 * P * P - 0.2 * P * g.adj(P) * P)


# training data: the loop sums of all four directions of one gauge field
U = g.qcd.gauge.random(grid, rng, scale=0.5)
P = [loop_sum(U, mu, 0.1) for mu in range(4)]
T = [teacher(x) for x in P]
identity = g.identity(P[0])

net = covariant_matrix_network(n_channels=4, n_layers=2)
wrng = np.random.default_rng(13)
w0 = net.weights(wrng)
g.message(f"Network with {len(w0)} complex weights")


# (1) gauge covariance
V = rng.element(g.mcolor(grid))
f0 = net(P[0], w0, identity)
f1 = net(g(V * P[0] * g.adj(V)), w0, identity)
eps2 = g.norm2(f1 - V * f0 * g.adj(V)) / g.norm2(f0)
g.message(f"Gauge covariance: {eps2}")
assert eps2 < 1e-28


# the loss graph, built once and re-used with new weight values:
#   loss = sum_mu |F(f(P_mu)) - F(T_mu)|^2 / sum_mu |F(T_mu) - F(P_mu)|^2
# (normalized to the non-trivial part of the teacher: 1 for f = identity)
# with F = identity or F = traceless_anti_hermitian (the update only sees
# TA(f), so the hermitian and trace parts of f are irrelevant there)
def build_loss(project):
    F = g.qcd.gauge.project.traceless_anti_hermitian if project else (lambda x: x)
    nw = [rad.node(x) for x in w0]
    nid = rad.node(identity, with_gradient=False)
    norm = sum(g.norm2(g(F(t) - F(p))) for p, t in zip(P, T))
    loss = None
    for p, t in zip(P, T):
        np_ = rad.node(p, with_gradient=False)
        nt = rad.node(g(F(t)), with_gradient=False)
        term = g.norm2(F(net(np_, nw, nid)) - nt)
        loss = term if loss is None else loss + term
    return loss * (1.0 / norm), nw


def evaluate(loss, nw, w, with_gradients=True):
    for n, x in zip(nw, w):
        n.value = x
    # the backward keeps the root value, which the next forward would reuse
    loss.value = None
    v = loss(with_gradients=with_gradients).real
    return v, ([n.gradient for n in nw] if with_gradients else None)


# (2) gradient w.r.t. the weights: the leaf gradient of a complex weight is
# dloss/dRe + i dloss/dIm, so the derivative along dw is Re(conj(gr) dw)
for project in [False, True]:
    loss, nw = build_loss(project)
    v0, gr = evaluate(loss, nw, w0)
    dw = net.weights(wrng, scale=1.0)
    a = sum((x.conjugate() * d).real for x, d in zip(gr, dw))
    h = 1e-4
    vp = [evaluate(loss, nw, [x + s * h * d for x, d in zip(w0, dw)], False)[0] for s in [2, 1, -1, -2]]
    b = (-vp[0] + 8 * vp[1] - 8 * vp[2] + vp[3]) / (12 * h)
    eps = abs(a - b) / abs(b)
    g.message(f"Gradient (TA loss = {project}): {a} vs {b}, rel {eps}")
    assert eps < 1e-8


# (3) the functional of the loss: parameters are found by identity, so equal
# weight values do not get confused (beta_0 is set to the value of a_00)
loss, nw = build_loss(True)
f = loss.functional(*nw)
w = list(w0)
j = 2 * net.C * net.C
w[j] = complex(w[0].real, w[0].imag)
assert w[j] == w[0] and w[j] is not w[0]
gr = f.gradient(w, w)
loss, nw = build_loss(True)
_, gr_ref = evaluate(loss, nw, w)
eps2 = sum(abs(x - y) ** 2 for x, y in zip(gr, gr_ref)) / sum(abs(y) ** 2 for y in gr_ref)
g.message(f"Functional gradient with equal weight values: {eps2}")
assert eps2 < 1e-28 and abs(gr[0] - gr[j]) > 1e-6


# (4) training with Adam, for the plain and the projected loss
n_iter = 300
for project in [False, True]:
    loss, nw = build_loss(project)
    f = loss.functional(*nw)
    w = list(w0)
    opt = g.algorithms.optimize.adam(
        maxiter=n_iter, alpha=5e-3, eps=1e-15, log_functional_every=30
    )
    opt(f)(w, w)
    initial, final = f(w0), f(w)
    g.message(f"TA loss = {project}: loss {initial} -> {final}")
    assert final < 1e-3 * initial
