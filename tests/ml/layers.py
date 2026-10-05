#!/usr/bin/env python3
#
# The basic layers of g.ml.layer (see also local_covariant_matrix.py)
#
import gpt as g
import numpy as np

rad = g.ad.reverse
rng = g.random("test")
grid = g.grid([4, 4, 4, 8], g.double)

# polynomial: y = x + c_2 x^2 + c_3 x^3, the identity at c = 0, covariant
P = g.mcolor(grid)
rng.normal_element(P)
P = g(0.3 * P)
f = g.ml.layer.polynomial(P, 3)
f.initialize(rng)
assert f["c"] == [0j, 0j] and g.norm2(f([P])[0] - P) == 0.0
f["c"] = [0.2 - 0.1j, -0.3 + 0.05j]
c2, c3 = f["c"]
(y,) = f([P])
eps = g.norm2(y - g(P + c2 * P * P + c3 * P * P * P)) / g.norm2(y)
g.message(f"polynomial value: {eps}")
assert eps < 1e-28
V = g.mcolor(grid)
rng.element(V)
(yV,) = f([g(V * P * g.adj(V))])
eps = g.norm2(yV - g(V * y * g.adj(V))) / g.norm2(y)
g.message(f"polynomial covariance: {eps}")
assert eps < 1e-28

# node vs plain evaluation, and the gradient w.r.t. the input and c
T = g.mcolor(grid)
rng.normal_element(T)
leaves = [rad.node(w) for w in f.parameters()]
x = rad.node(P)
(yn,) = f([x], leaves)
loss = g.norm2(yn - rad.node(T, with_gradient=False)).functional(x, *leaves)
eps = abs(loss([P] + list(f.parameters())) - g.norm2(y - T)) / g.norm2(y - T)
g.message(f"polynomial node vs plain: {eps}")
assert eps < 1e-14
loss.assert_gradient_error(rng, [P] + list(f.parameters()), [P] + list(f.parameters()), 1e-4, 1e-8)

# a covariant-word model with MLP coefficients:
# f(P) = P + sum_w c_w(I(P)) w(P), invariants -> mlp -> matrix_words, on a
# general (not unitary) matrix field
P = g.mcolor(grid)
rng.cnormal(P, sigma=0.3)
(x,) = g.ml.symbols("P")
inv = g.ml.layer.matrix_invariants(P)
(I,) = inv([x], name="invariants")
(c,) = g.ml.layer.mlp(P, inv.n, g.ml.layer.matrix_words.n, width=4, depth=2)([I], name="mlp")
(fP,) = g.ml.layer.matrix_words(P)([x, c], name="words")
net = g.ml.pack(f=fP).function()
net.initialize(rng)
net.calibrate([[P]])
(I_P,) = inv([P])
for k, q in enumerate(I_P):
    m = g.sum(q).real / grid.gsites
    s = g.sum(g(q * q)).real / grid.gsites
    assert abs(m) < 1e-12 and abs(s - 1) < 1e-10, (k, m, s)
# the identity at initialization
assert g.norm2(net([P])[0] - P) == 0.0
# a nonzero output layer: covariance and gradients
for i, w in enumerate(net.parameters()):
    if net.parameter_names()[i].startswith("mlp.W2") or net.parameter_names()[i].startswith("mlp.b2"):
        net.parameters()[i] = 0.1 * rng.normal_element(0j)
(y,) = net([P])
(yV,) = net([g(V * P * g.adj(V))])
eps = g.norm2(yV - g(V * y * g.adj(V))) / g.norm2(y)
g.message(f"covariant words: covariance {eps}, |f - P|^2 / |P|^2 = {g.norm2(y - P) / g.norm2(P)}")
assert eps < 1e-26 and g.norm2(y - P) > 0
leaves = [rad.node(w) for w in net.parameters()]
x = rad.node(P)
(yn,) = net([x], leaves)
loss = g.norm2(yn - rad.node(T, with_gradient=False)).functional(x, *leaves)
fields = [P] + list(net.parameters())
loss.assert_gradient_error(rng, fields, fields, 1e-5, 1e-7)
