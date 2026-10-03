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

# by default the training runs only a few steps (and checks that the loss
# decreases); --stringent runs the full training (about a minute) with the
# assertions on the reached losses
stringent = g.default.has("--stringent")
n_iter = 300 if stringent else 30
grid = g.grid([4, 4, 4, 4], g.double)


class covariant_matrix_network:
    # C channels X_c of N x N matrices (all transforming like P), L residual
    # layers
    #   Y_c = sum_d (a_cd X_d + b_cd X_d^dag) + beta_c 1
    #   Z_c = Y_c Y_{c+1 mod C}
    #   Z_c <- relu(alpha_c (q_c - qmean_c) / qstd_c + gamma_c) Z_c   (optional gate)
    #   X_c <- X_c + Z_c
    # with the invariant q_c = tr(Y_c Y_c^dag) / N, and the output
    # f = P + sum_c w_c X_c, starting from X_c = P.  Weights are complex
    # scalars; every operation is covariant (scalar coefficients, products,
    # adjoints, the identity, and gates that are functions of gauge
    # invariants).  relu acts on complex numbers as z for Re z > 0 and 0
    # otherwise.  The same code runs on plain fields (weights as numbers) and
    # on nodes (weights as node leaves).
    #
    # The gate references qmean_c, qstd_c are constants: the mean and the
    # standard deviation of q_c over the sites of the training data at the
    # initial weights (see calibrate), so that the gate argument varies by
    # O(alpha) between sites.  They are frozen (not live lattice averages) to
    # keep f site-local.  For Y_c = lambda P the standardized invariant does
    # not depend on lambda, so a gate can express a threshold in tr(P P^dag).
    def __init__(self, n_channels, n_layers, gate=False):
        self.C = n_channels
        self.L = n_layers
        self.gate = gate
        self.n_layer = 2 * n_channels * n_channels + n_channels + (2 * n_channels if gate else 0)
        self.q_ref = None

    def calibrate(self, P, w, identity, one):
        # layer by layer, as each layer's references enter the later layers
        if not self.gate:
            return
        self.q_ref = []
        for layer in range(self.L):
            q = [self(p, w, identity, one, measure=layer) for p in P]
            self.q_ref.append([standardization([x[c] for x in q]) for c in range(self.C)])

    def weights(self, rng, scale=0.1):
        C = self.C

        def r():
            return complex(scale * (rng.normal() + 1j * rng.normal()) / np.sqrt(2))

        w = []
        for _ in range(self.L):
            w += [r() for _ in range(2 * C * C + C)]
            if self.gate:
                # gates start close to 1, away from the kink of relu
                w += [r() for _ in range(C)] + [1.0 + r() for _ in range(C)]
        # a small output: f starts close to the identity map (with zero output
        # weights the inner weights would get no gradient)
        w += [0.1 * r() for _ in range(C)]
        return w

    def __call__(self, P, w, identity, one, measure=None):
        # identity: the unit matrix field, one: the unit complex field;
        # measure = layer: return the invariants q_c of that layer (plain
        # fields only)
        C = self.C
        N = P.otype.shape[0]
        X = [P] * C
        k = 0
        for layer in range(self.L):
            Xa = [g.adj(x) for x in X]
            Y = []
            for c in range(C):
                y = w[k + 2 * C * C + c] * identity
                for d in range(C):
                    y = y + w[k + c * C + d] * X[d] + w[k + C * C + c * C + d] * Xa[d]
                Y.append(y)
            Z = [Y[c] * Y[(c + 1) % C] for c in range(C)]
            if self.gate:
                kg = k + 2 * C * C + C
                q = [g(g.trace(Y[c] * g.adj(Y[c])) * (1.0 / N)) for c in range(C)]
                if measure == layer:
                    return q
                for c in range(C):
                    alpha, gamma = w[kg + c], w[kg + C + c]
                    mean, std = self.q_ref[layer][c]
                    # alpha (q - mean) / std + gamma
                    s = g(alpha * q[c] * (1.0 / std) + (gamma - alpha * (mean / std)) * one)
                    Z[c] = g.component.relu()(s) * Z[c]
            X = [g(X[c] + Z[c]) for c in range(C)]
            k += self.n_layer
        f = P
        for c in range(C):
            f = f + w[k + c] * X[c]
        return g(f)


def standardization(q):
    # mean and standard deviation over the sites of a list of real-valued
    # complex fields
    n = sum(x.grid.gsites for x in q)
    mean = sum(g.sum(x).real for x in q) / n
    mean2 = sum(g.sum(g(x * x)).real for x in q) / n
    return mean, np.sqrt(mean2 - mean**2)


def loop_sum(U, mu, rho):
    # rho * sum_nu (plaquettes at x through the link (x, mu), ending with b(mu))
    def staples(nu):
        U_mu_nu = g.cshift(U[mu], nu, 1)
        up = U[nu] * U_mu_nu * g.adj(g.cshift(U[nu], mu, 1))
        U_nu_m = g.cshift(U[nu], nu, -1)
        down = g.adj(U_nu_m) * g.cshift(U[mu], nu, -1) * g.cshift(U_nu_m, mu, 1)
        return g(up + down)

    staple = g(sum(staples(nu) for nu in range(len(U)) if nu != mu))
    return g(rho * staple * g.adj(U[mu]))


def teacher(P):
    return g(P + 0.3 * P * P - 0.2 * P * g.adj(P) * P)


def invariant(P):
    return g(g.trace(P * g.adj(P)) * (1.0 / P.otype.shape[0]))


def teacher_threshold(P, mean, std, c=0.1):
    # a P^2 term that acts only where tr(P P^dag) is above its mean (in
    # units of its standard deviation over the training data)
    r = g.component.relu()(g(invariant(P) * (1.0 / std) - (mean / std) * one))
    return g(P + c * r * P * P)


# training data: the loop sums of all four directions of one gauge field
U = g.qcd.gauge.random(grid, rng, scale=0.5)
P = [loop_sum(U, mu, 0.1) for mu in range(4)]
T = [teacher(x) for x in P]
identity = g.identity(P[0])
one = g.complex(P[0].grid)
one[:] = 1
q_mean, q_std = standardization([invariant(x) for x in P])
T_threshold = [teacher_threshold(x, q_mean, q_std) for x in P]
wrng = np.random.default_rng(13)
nets = {gate: covariant_matrix_network(n_channels=4, n_layers=2, gate=gate) for gate in [False, True]}
weights = {gate: nets[gate].weights(wrng) for gate in nets}
for gate, net in nets.items():
    net.calibrate(P, weights[gate], identity, one)


# (1) gauge covariance
V = rng.element(g.mcolor(grid))
for gate, net in nets.items():
    w0 = weights[gate]
    g.message(f"Network (gate = {gate}) with {len(w0)} complex weights")
    f0 = net(P[0], w0, identity, one)
    f1 = net(g(V * P[0] * g.adj(V)), w0, identity, one)
    eps2 = g.norm2(f1 - V * f0 * g.adj(V)) / g.norm2(f0)
    g.message(f"Gauge covariance (gate = {gate}): {eps2}")
    assert eps2 < 1e-28


# the loss graph, built once and re-used with new weight values:
#   loss = sum_mu |F(f(P_mu)) - F(T_mu)|^2 / sum_mu |F(T_mu) - F(P_mu)|^2
# (normalized to the non-trivial part of the teacher: 1 for f = identity)
# with F = identity or F = traceless_anti_hermitian (the update only sees
# TA(f), so the hermitian and trace parts of f are irrelevant there)
def build_loss(net, w0, project, T=T):
    F = g.qcd.gauge.project.traceless_anti_hermitian if project else (lambda x: x)
    nw = [rad.node(x) for x in w0]
    nid = rad.node(identity, with_gradient=False)
    none = rad.node(one, with_gradient=False)
    norm = sum(g.norm2(g(F(t) - F(p))) for p, t in zip(P, T))
    loss = sum(
        g.norm2(
            F(net(rad.node(p, with_gradient=False), nw, nid, none))
            - rad.node(g(F(t)), with_gradient=False)
        )
        for p, t in zip(P, T)
    )
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
for gate, net in nets.items():
    w0 = weights[gate]
    for project in [False, True]:
        loss, nw = build_loss(net, w0, project)
        v0, gr = evaluate(loss, nw, w0)
        dw = net.weights(wrng, scale=1.0)
        a = sum((x.conjugate() * d).real for x, d in zip(gr, dw))
        h = 1e-4
        vp = [
            evaluate(loss, nw, [x + s * h * d for x, d in zip(w0, dw)], False)[0]
            for s in [2, 1, -1, -2]
        ]
        b = (-vp[0] + 8 * vp[1] - 8 * vp[2] + vp[3]) / (12 * h)
        eps = abs(a - b) / abs(b)
        g.message(f"Gradient (gate = {gate}, TA loss = {project}): {a} vs {b}, rel {eps}")
        assert eps < 1e-8


# (3) the functional of the loss: parameters are found by identity, so equal
# weight values do not get confused (beta_0 is set to the value of a_00)
net, w0 = nets[False], weights[False]
loss, nw = build_loss(net, w0, True)
f = loss.functional(*nw)
w = list(w0)
j = 2 * net.C * net.C
w[j] = complex(w[0].real, w[0].imag)
assert w[j] == w[0] and w[j] is not w[0]
gr = f.gradient(w, w)
loss, nw = build_loss(net, w0, True)
_, gr_ref = evaluate(loss, nw, w)
eps2 = sum(abs(x - y) ** 2 for x, y in zip(gr, gr_ref)) / sum(abs(y) ** 2 for y in gr_ref)
g.message(f"Functional gradient with equal weight values: {eps2}")
assert eps2 < 1e-28 and abs(gr[0] - gr[j]) > 1e-6


# (4) training with Adam, for the plain and the projected loss
for gate, net in nets.items():
    w0 = weights[gate]
    for project in [False, True]:
        loss, nw = build_loss(net, w0, project)
        f = loss.functional(*nw)
        w = list(w0)
        opt = g.algorithms.optimize.adam(
            maxiter=n_iter, alpha=5e-3, eps=1e-15, log_functional_every=100
        )
        opt(f)(w, w)
        initial, final = f(w0), f(w)
        g.message(f"Gate = {gate}, TA loss = {project}: loss {initial} -> {final}")
        assert final < (1e-3 if stringent else 1e-1) * initial


# (5) the threshold teacher: the gated network represents it exactly with
# Y_c = X_c, the gate relu((q_0 - mean) / std) in channel 0 of layer 0, all
# other gates zero, and the output f = P + c (X_0 - X_1)
net = covariant_matrix_network(n_channels=4, n_layers=2, gate=True)
C = net.C
w = [0.0] * len(weights[True])
for layer in range(net.L):
    k = layer * net.n_layer
    for c in range(C):
        w[k + c * C + c] = 1.0
w[2 * C * C + C] = 1.0
w[net.L * net.n_layer] = 0.1
w[net.L * net.n_layer + 1] = -0.1
w = [complex(x) for x in w]
net.calibrate(P, w, identity, one)
loss, nw = build_loss(net, w, False, T_threshold)
eps = loss.functional(*nw)(w)
g.message(f"Gated network at the exact threshold solution: loss {eps}")
assert eps < 1e-20

# the gates help, but training does not find the exact solution: most gates
# end up off (or on) on every site and act as channel switches instead of
# thresholds; the comparison below is for 300 steps (--stringent)
final = {}
for gate, net in nets.items():
    w0 = weights[gate]
    loss, nw = build_loss(net, w0, False, T_threshold)
    f = loss.functional(*nw)
    w = list(w0)
    opt = g.algorithms.optimize.adam(
        maxiter=n_iter, alpha=5e-3, eps=1e-15, log_functional_every=100
    )
    opt(f)(w, w)
    final[gate] = f(w)
    g.message(f"Threshold teacher, gate = {gate}: loss {f(w0)} -> {final[gate]}")
    assert final[gate] < f(w0)
if stringent:
    assert final[True] < 0.8 * final[False]
