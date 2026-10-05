#!/usr/bin/env python3
#
# Diagnostics of g.ml functions and their training: snapshot / displacement,
# activity of the calls of a composite, gradient noise of a stochastic cost.
#
import gpt as g
import numpy as np

rad = g.ad.reverse
rng = g.random("test")
grid = g.grid([4, 4, 4, 8], g.double)
P = g.mcolor(grid)
rng.normal_element(P)
P = g(0.5 * P)

# replicate -> local_covariant_matrix -> linear_combination
(x,) = g.ml.symbols("P")
(X,) = g.ml.layer.replicate(P, 2)([x])
(X,) = g.ml.layer.local_covariant_matrix(P, 2)([X], name="block")
(y,) = g.ml.layer.linear_combination(P, 2)([x, X], name="readout")
net = g.ml.pack(f=y).function()

# activity: at scale 0 every call is the identity in its input; the replicate
# has no input of the type of its output (a list)
net.initialize(rng, scale=0.0)
a = g.ml.activity(net, [P])
g.message(f"activity at scale 0: {a}")
assert a["block"] == 0.0 and a["readout"] == 0.0 and a[""] == 0.0
assert a["replicate"] is None

# with a gain, the block's ratio is |sum_c gain_c Z_c| / |X| and the readout's
# |sum_c w_c X_c| / |P|
net.initialize(rng)
a = g.ml.activity(net, [P])
g.message(f"activity: {a}")
X0 = [P, P]
(X1,) = net._functions["block"]([X0])
ref = (sum(g.norm2(u - v) for u, v in zip(X1, X0)) / (2 * g.norm2(P))) ** 0.5
assert abs(a["block"] - ref) < 1e-10 * ref
(y1,) = net([P])
ref = (g.norm2(y1 - P) / g.norm2(P)) ** 0.5
assert abs(a["readout"] - ref) < 1e-10 * ref and abs(a[""] - ref) < 1e-10 * ref
a = g.ml.activity(net._functions["block"], [X0])
assert list(a.keys()) == [""]

# displacement
reference = g.ml.snapshot(net)
d = g.ml.displacement(net, reference)
assert all(v[0] == 0.0 for v in d.values())
w0 = net["readout.w.1"]
net["readout.w.1"] = w0 + 0.1j
d = g.ml.displacement(net, reference)
g.message(f"displacement of readout.w.1: {d['readout.w.1']}")
assert abs(d["readout.w.1"][0] - 0.1) < 1e-14
assert abs(d["readout.w.1"][1] - 0.1 / abs(w0)) < 1e-12
assert all(v[0] == 0.0 for k, v in d.items() if k != "readout.w.1")
net["block.gain"] = [0j, 0j]
assert g.ml.displacement(net, g.ml.snapshot(net))["block.gain.0"][1] is None

# gradient noise: |<v, f(P) - T>|^2 for random directions v is an unbiased
# estimate of |f(P) - T|^2 / N_dof times a constant; its gradient is noisy,
# the deterministic cost |f(P) - T|^2 is not
net.initialize(rng)
T = g.mcolor(grid)
rng.normal_element(T)
weights = net.parameters()
leaves = [rad.node(w) for w in weights]
(yn,) = net([rad.node(P, with_gradient=False)], leaves)
d_exact = g.norm2(yn - rad.node(T, with_gradient=False)).functional(*leaves)
names = net.parameter_names()
noise = g.ml.gradient_noise(lambda: d_exact, weights, 4, names)
assert all(v[1] < 1e-10 * v[0] and v[2] > 1e18 for v in noise.values())


def stochastic():
    v = g.mcolor(grid)
    rng.normal_element(v)
    v = rad.node(v, with_gradient=False)
    (yn,) = net([rad.node(P, with_gradient=False)], leaves)
    q = g.inner_product(v, yn - rad.node(T, with_gradient=False))
    return (q * g.adj(q)).functional(*leaves)


noise = g.ml.gradient_noise(stochastic, weights, 8, names)
for name in ["block.gain.0", "readout.w.0"]:
    g.message(
        f"gradient noise {name}: |mean| {noise[name][0]:.3e} std {noise[name][1]:.3e} snr {noise[name][2]:.3e}"
    )
assert all(0 < v[1] and v[2] < 1e3 for v in noise.values())
