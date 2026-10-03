#!/usr/bin/env python3
#
# A learnable loop function: a g.ml network f(P) with global number weights
# inside two layers of directional_parallel_transport, U_mu' =
# exp(TA(P1 f(P))) U_mu with P = rho sum(plaquettes) and a trained global
# real rho, trained on squared force contractions
# Q = <v, F> of the combined action (gauge action of the flowed links and
# log det of the Jacobians), F = dS/dU, with fixed directions v.
#
import gpt as g

rad = g.ad.reverse
rng = g.random("test")
# (8 in time: with --mpi 1.1.1.2 the local extent must allow SIMD and checkerboarding)
grid = g.grid([4, 4, 4, 8], g.double)

# by default a few training steps (the loss decreases); --stringent: more
stringent = g.default.has("--stringent")
n_iter = 20 if stringent else 3

U = g.qcd.gauge.random(grid, rng, scale=0.5)
even, odd = g.even_odd_projectors(grid)
full = g(even + odd)
mu = 0

# rho: a global real number (g.ml.layer.broadcast), which enters the
# transport as its field-valued weight
rho_fn = g.ml.layer.broadcast(g.complex(grid), value=0.1, real=True)
rho_fn.initialize(rng)
(rho,) = rho_fn([])
description = [(rho, g.path().f(nu).f(mu).b(nu).b(mu)) for nu in range(4) if nu != mu] + [
    (rho, g.path().b(nu).f(mu).f(nu).b(mu)) for nu in range(4) if nu != mu
]


def loop_sum(U):
    # the argument of the loop function: rho * sum of the plaquettes through
    # the link (x, mu), as the transport computes it
    pt, info = g.parallel_transport_weighted(U, description)
    return g(sum(x if w is None else w * x for (_, w), x in zip(info, pt(U))))


# the learnable function: P -> 2 channels -> one block -> P + sum_c w_c X_c
P = loop_sum(U)
(x,) = g.ml.symbols("P")
(X,) = g.ml.layer.replicate(P, 2)([x])
(X,) = g.ml.layer.local_covariant_matrix(P, 2)([X], name="block")
(y,) = g.ml.layer.linear_combination(P, 2)([x, X], name="readout")
net = g.ml.pack(f=y).function()
net.initialize(rng)
n_par = len(net.parameters())
g.message(f"Loop function with {n_par} global number weights: {net.parameter_names()}")

# two flow layers (odd, then even sites) sharing the loop function; the
# transport parameters are rho and the network's weights
params = [rho] + list(net.parameters())


def layer(P1):
    return g.qcd.gauge.smear.directional_parallel_transport(
        U, description, mu, full, P1, params, loop_function=lambda sm, xp: net([sm], xp[1:])[0]
    )


phi_1, phi_2 = layer(odd), layer(even)
indices = list(range(5 + n_par))
a_gauge = g.qcd.gauge.action.iwasaki(6)
a_gauge = a_gauge.transformed(phi_2, indices=indices, projection=[0, 1, 2, 3])
a_gauge = a_gauge.transformed(phi_1, indices=indices, projection=indices)
a_ld_2 = phi_2.action_log_det_jacobian().transformed(phi_1, indices=indices, projection=indices)
S = a_gauge + phi_1.action_log_det_jacobian() + a_ld_2

# the combined action and its gradient w.r.t. links, the rho field and the
# weights; g.ml.fields makes one list of them (with the network's own storage)
fields = g.ml.fields(U, [rho], net.parameters())
S.assert_gradient_error(rng, fields, fields, 1e-4, 1e-7)

# the loss: sum_i Q_i^2 with Q_i = <v_i, F>, as a node graph over the weights;
# the gradient of each Q_i w.r.t. the weights is the difference of the
# weight-gradient of S along v_i (g.group.directional_derivative)
Qs = [
    g.group.directional_derivative(S, rng.normal_element(g.group.cartesian(U)), along=[0, 1, 2, 3])
    for _ in range(2)
]
# the trained parameters: rho (a global number, broadcast to the transport's
# field) and the weights, in their own storage
weights = g.ml.fields(rho_fn.parameters(), net.parameters())
leaves = [rad.node(w) for w in weights]
(rho_node,) = rho_fn([], leaves[0:1])
loss = sum(q * q for q in [rad.functional_node(Q, U + [rho_node] + leaves[1:]) for Q in Qs])
loss = loss.functional(*leaves)
loss.assert_gradient_error(rng, weights, weights, 1e-4, 1e-6)
loss.assert_gradient_error(rng, weights, [weights[0]], 1e-4, 1e-6)

# training: the optimizer updates rho and the weights in their storage (with
# alpha = 1e-2 the weights reach a nearly singular Jacobian after ~15 steps
# and the loss explodes)
initial = loss(weights)
opt = g.algorithms.optimize.adam(maxiter=n_iter, alpha=5e-3, eps=1e-15, log_functional_every=1)
opt(loss)(weights, weights)
final = loss(weights)
g.message(f"Training: loss {initial} -> {final}, rho = {rho_fn['value']}")
assert final < initial
assert rho_fn["value"].imag == 0.0 and rho_fn["value"] != 0.1
