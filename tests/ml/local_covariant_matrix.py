#!/usr/bin/env python3
#
# Training a gauge-covariant function f(P) of a site-local matrix field P,
# P(x) -> V(x) P(x) V(x)^dag, with f(V P V^dag) = V f(P) V^dag, built from
# g.ml.layer.local_covariant_matrix blocks.  P is the weighted plaquette loop
# sum through a link, the argument of loop_function in
# directional_parallel_transport.
#
import gpt as g
import numpy as np

rad = g.ad.reverse
rng = g.random("test")
grid = g.grid([4, 4, 4, 4], g.double)

# by default the training runs only a few steps (and checks that the loss
# decreases); --stringent runs the full training (a few minutes) with the
# assertions on the reached losses
stringent = g.default.has("--stringent")
n_iter = 300 if stringent else 30


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


def invariant(P):
    return g(g.trace(P * g.adj(P)) * (1.0 / P.otype.shape[0]))


# training data: the loop sums of all four directions of one gauge field
U = g.qcd.gauge.random(grid, rng, scale=0.5)
P = [loop_sum(U, mu, 0.1) for mu in range(4)]
one = g.complex(grid)
one[:] = 1

# teachers: a polynomial, and a P^2 term acting only where tr(P P^dag) is
# above its mean (in units of its standard deviation over the data)
q = [invariant(x) for x in P]
n = sum(x.grid.gsites for x in q)
q_mean = sum(g.sum(x).real for x in q) / n
q_std = np.sqrt(sum(g.sum(g(x * x)).real for x in q) / n - q_mean**2)
T = [g(x + 0.3 * x * x - 0.2 * x * g.adj(x) * x) for x in P]
T_threshold = [
    g(x + 0.1 * g.component.relu()(g(qx * (1.0 / q_std) - (q_mean / q_std) * one)) * x * x)
    for x, qx in zip(P, q)
]


def network(n_channels, n_layers, gate):
    # P -> C channels -> n_layers blocks -> f = P + sum_c w_c X_c
    (x,) = g.ml.symbols("P")
    (X,) = g.ml.layer.replicate(P[0], n_channels)([x])
    for layer in range(n_layers):
        (X,) = g.ml.layer.local_covariant_matrix(P[0], n_channels, gate)([X], name=f"layer{layer}")
    (f,) = g.ml.layer.linear_combination(P[0], n_channels)([x, X], name="readout")
    net = g.ml.pack(f=f).function()
    net.initialize(rng)
    net.calibrate([[x] for x in P])
    return net


nets = {gate: network(4, 2, gate) for gate in [False, True]}
initial = {gate: g.ml.snapshot(net) for gate, net in nets.items()}


def reset(net, values):
    for name, v in values.items():
        net[name] = v


# (1) gauge covariance
V = rng.element(g.mcolor(grid))
for gate, net in nets.items():
    g.message(f"Network (gate = {gate}) with {len(net.parameters())} complex weights")
    f0 = net([P[0]])[0]
    f1 = net([g(V * P[0] * g.adj(V))])[0]
    eps2 = g.norm2(f1 - V * f0 * g.adj(V)) / g.norm2(f0)
    g.message(f"Gauge covariance (gate = {gate}): {eps2}")
    assert eps2 < 1e-28


# the loss functional over the network's parameters:
#   loss = sum_mu |F(f(P_mu)) - F(T_mu)|^2 / sum_mu |F(T_mu) - F(P_mu)|^2
# (normalized to the non-trivial part of the teacher: 1 for f = identity)
# with F = identity or F = traceless_anti_hermitian (the update only sees
# TA(f), so the hermitian and trace parts of f are irrelevant there)
def loss(net, project, T=T):
    F = g.qcd.gauge.project.traceless_anti_hermitian if project else (lambda x: x)
    leaves = [rad.node(x) for x in net.parameters()]
    norm = sum(g.norm2(g(F(t) - F(p))) for p, t in zip(P, T))
    terms = sum(
        g.norm2(F(net([rad.node(p, with_gradient=False)], leaves)[0]) - rad.node(g(F(t)), with_gradient=False))
        for p, t in zip(P, T)
    )
    return (terms * (1.0 / norm)).functional(*leaves)


# (2) gradient w.r.t. the weights
for gate, net in nets.items():
    for project in [False, True]:
        g.message(f"Gradient (gate = {gate}, TA loss = {project})")
        loss(net, project).assert_gradient_error(rng, net.parameters(), net.parameters(), 1e-4, 1e-8)


# the functional finds parameters by identity: equal values are different
# parameters (beta_0 of layer 0 is set to the value of a_00)
net = nets[False]
f = loss(net, True)
w = list(net.parameters())
j = net.parameter_names().index("layer0.beta.0")
w[j] = complex(w[0].real, w[0].imag)
assert w[j] == w[0] and w[j] is not w[0]
gr = f.gradient(w, w)
eps = abs(gr[j] - f.gradient(w, [w[j]])[0]) / abs(gr[j])
g.message(f"Gradient with equal weight values: {eps}")
assert eps < 1e-14 and abs(gr[0] - gr[j]) > 1e-6


# (3) training with Adam, for the plain and the projected loss
def train(gate, f):
    net = nets[gate]
    reset(net, initial[gate])
    opt = g.algorithms.optimize.adam(maxiter=n_iter, alpha=5e-3, eps=1e-15, log_functional_every=100)
    start = f(net.parameters())
    opt(f)(net.parameters(), net.parameters())
    return start, f(net.parameters())


for gate, net in nets.items():
    for project in [False, True]:
        start, final = train(gate, loss(net, project))
        g.message(f"Gate = {gate}, TA loss = {project}: loss {start} -> {final}")
        assert final < (1e-3 if stringent else 1e-1) * start


# (4) the threshold teacher: the gated network represents it exactly with
# Y_c = X_c, the gate relu((q_0 - mean) / std) in channel 0 of layer 0, all
# other gates zero, and the output f = P + c (X_0 - X_1)
net = network(4, 2, True)
C = 4
for layer in range(2):
    net[f"layer{layer}.a"] = [complex(c == d) for c in range(C) for d in range(C)]
    for slot in ["b", "beta", "alpha", "gamma"]:
        net[f"layer{layer}.{slot}"] = [0j] * len(net[f"layer{layer}.{slot}"])
    net[f"layer{layer}.gain"] = [1 + 0j] * C
net["layer0.alpha.0"] = 1 + 0j
net["readout.w"] = [0.1 + 0j, -0.1 + 0j, 0j, 0j]
net.calibrate([[x] for x in P])
eps = loss(net, False, T_threshold)(net.parameters())
g.message(f"Gated network at the exact threshold solution: loss {eps}")
assert eps < 1e-20

# the gates help, but training does not find the exact solution: most gates
# end up off (or on) on every site and act as channel switches instead of
# thresholds; the comparison below is for 300 steps (--stringent)
final = {}
for gate, net in nets.items():
    start, final[gate] = train(gate, loss(net, False, T_threshold))
    g.message(f"Threshold teacher, gate = {gate}: loss {start} -> {final[gate]}")
    assert final[gate] < start
if stringent:
    assert final[True] < 0.8 * final[False]
