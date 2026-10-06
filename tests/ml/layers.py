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
# a nonzero output layer (an array, the last column the bias): covariance and gradients
net["mlp.W2"] = 0.1 * rng.normal_element(np.zeros_like(net["mlp.W2"]))
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

# invariants of fixed loops as further inputs: a loop function f(P, L) of
# directional_parallel_transport with fixed loops (loops=...), invariants of
# P and of the loops L_k -> mlp -> matrix_words(P)
n_loops = 2
L = [g.mcolor(grid) for _ in range(n_loops)]
rng.element(L)
x, l = g.ml.symbols("P", "L")
inv_l = g.ml.layer.matrix_invariants(P, n_loops)
assert inv_l.n == 5 + 2 * n_loops and inv.n == 5
(I,) = inv_l([x, l], name="invariants")
(c,) = g.ml.layer.mlp(P, inv_l.n, g.ml.layer.matrix_words.n, width=4, depth=2)([I], name="mlp")
(fP,) = g.ml.layer.matrix_words(P)([x, c], name="words")
net_l = g.ml.pack(f=fP).function()
net_l.initialize(rng)
net_l.calibrate([[P, L]])
(I_P,) = inv_l([P, L])
for k, q in enumerate(I_P):
    m = g.sum(q).real / grid.gsites
    s = g.sum(g(q * q)).real / grid.gsites
    assert abs(m) < 1e-12 and abs(s - 1) < 1e-10, (k, m, s)
# loop_imag=False: Re tr L_k / 3 only, the same as the corresponding entries
inv_r = g.ml.layer.matrix_invariants(P, n_loops, loop_imag=False)
assert inv_r.n == 5 + n_loops
inv_r.calibrate([[P, L]])
(I_r,) = inv_r([P, L])
for j, k in enumerate(list(range(5)) + [5 + 2 * i for i in range(n_loops)]):
    assert g.norm2(I_r[j] - I_P[k]) < 1e-24 * max(g.norm2(I_P[k]), 1.0)
# the raw loop invariants (Re, Im tr L_k / 3)
for k, L_k in enumerate(L):
    t = g(g.trace(L_k) * (1.0 / 3.0))
    for j, part in enumerate([g.component.real(t), g.component.imag(t)]):
        i = 5 + 2 * k + j
        q = g(I_P[i] * (1.0 / inv_l["inv_std"][i]) + inv_l["mean"][i] * inv_l.one)
        assert g.norm2(q - part) < 1e-24 * g.norm2(part)
# the invariants of P agree with the layer without loops (calibrated alike)
(I_0,) = inv([P])
assert sum(g.norm2(a - b) for a, b in zip(I_0, I_P[0:5])) < 1e-24
assert g.norm2(net_l([P, L])[0] - P) == 0.0
net_l["mlp.W2"] = 0.1 * rng.normal_element(np.zeros_like(net_l["mlp.W2"]))
(y,) = net_l([P, L])
(yV,) = net_l([g(V * P * g.adj(V)), [g(V * L_k * g.adj(V)) for L_k in L]])
eps = g.norm2(yV - g(V * y * g.adj(V))) / g.norm2(y)
g.message(f"loop invariants: covariance {eps}")
assert eps < 1e-26
# the loops change the output
L2 = [g.mcolor(grid) for _ in range(n_loops)]
rng.element(L2)
assert g.norm2(net_l([P, L2])[0] - y) > 1e-6 * g.norm2(y)
# gradients w.r.t. P, the loops and the parameters
leaves = [rad.node(w) for w in net_l.parameters()]
x = rad.node(P)
l = [rad.node(L_k) for L_k in L]
(yn,) = net_l([x, l], leaves)
loss = g.norm2(yn - rad.node(T, with_gradient=False)).functional(x, *l, *leaves)
fields = [P] + L + list(net_l.parameters())
eps = abs(loss(fields) - g.norm2(y - T)) / g.norm2(y - T)
g.message(f"loop invariants: node vs plain {eps}")
assert eps < 1e-14
loss.assert_gradient_error(rng, fields, fields, 1e-5, 1e-7)

# as the loop function of a transport with the 2x1 rectangles around U_0(x)
# in the planes (0, 1) and (0, 2) as fixed loops: the log det and its
# gradient w.r.t. the links and the network's weights
U = g.qcd.gauge.random(grid, rng, scale=0.5)
even, odd = g.even_odd_projectors(grid)
rho = g.complex(grid)
rho[:] = 0.1
description = [(rho, g.path().f(nu).f(0).b(nu).b(0)) for nu in range(1, 4)] + [
    (rho, g.path().b(nu).f(0).f(nu).b(0)) for nu in range(1, 4)
]
loops = [g.path().f(nu).f(0).b(nu).b(nu).b(0).f(nu) for nu in range(1, 3)]
params = [rho] + list(net_l.parameters())
pt = g.qcd.gauge.smear.directional_parallel_transport(
    U,
    description,
    0,
    g(even + odd),
    odd,
    params,
    loop_function=lambda sm, xp, L: net_l([sm, L], xp[1:])[0],
    loops=loops,
)
fields = g.ml.fields(U, [rho], net_l.parameters())
pt.action_log_det_jacobian().assert_gradient_error(rng, fields, fields, 1e-4, 1e-7)
Up = pt(fields)
Uinv = pt.inv(Up[0:4] + params)
eps2 = g.norm2(Uinv[0] - U[0]) / g.norm2(U[0])
g.message(f"loop invariants: transport inverse {eps2}")
assert eps2 < 1e-25

# words linear in the loops: matrix_words(P, n_loops) with the words L_k,
# P L_k, L_k P (and L_k^dag), and the mixed invariants tr P L_k
Lw = [L[0], g(0.5 * (L[1] + g.adj(L[1])))]  # the second loop hermitian
words = g.ml.layer.matrix_words(P, n_loops, loop_adjoint=[True, False])
assert words.n == 6 + 4 + 3 and g.ml.layer.matrix_words.n == 6
cw = [g.complex(grid) for _ in range(words.n)]
for z in cw:
    rng.cnormal(z, sigma=0.1)
(y,) = words([P, cw, Lw])
Pd, P2 = g(g.adj(P)), g(P * P)
expected = [P2, Pd, g(P * Pd), g(Pd * P), g(P2 * P), g(P2 * Pd)]
expected += [Lw[0], g(P * Lw[0]), g(Lw[0] * P), g(g.adj(Lw[0])), Lw[1], g(P * Lw[1]), g(Lw[1] * P)]
y_ref = P
for z, w in zip(cw, expected):
    y_ref = g(y_ref + z * w)
eps = g.norm2(y - y_ref) / g.norm2(y_ref)
g.message(f"loop words: value {eps}")
assert eps < 1e-28

inv_w = g.ml.layer.matrix_invariants(P, n_loops, loop_imag=[True, False], mixed=True)
assert inv_w.n == 5 + 3 + 4
x, l = g.ml.symbols("P", "L")
(I,) = inv_w([x, l], name="invariants")
(c,) = g.ml.layer.mlp(P, inv_w.n, words.n, width=4, depth=2)([I], name="mlp")
(fP,) = g.ml.layer.matrix_words(P, n_loops, loop_adjoint=[True, False])([x, c, l], name="words")
net_w = g.ml.pack(f=fP).function()
net_w.initialize(rng)
net_w.calibrate([[P, Lw]])
# the raw mixed invariants (Re, Im tr P L_k / 3)
(I_w,) = inv_w([P, Lw])
for k, L_k in enumerate(Lw):
    t = g(g.trace(P * L_k) * (1.0 / 3.0))
    for j, part in enumerate([g.component.real(t), g.component.imag(t)]):
        i = 8 + 2 * k + j
        q = g(I_w[i] * (1.0 / inv_w["inv_std"][i]) + inv_w["mean"][i] * inv_w.one)
        assert g.norm2(q - part) < 1e-24 * g.norm2(part)
assert g.norm2(net_w([P, Lw])[0] - P) == 0.0
net_w["mlp.W2"] = 0.1 * rng.normal_element(np.zeros_like(net_w["mlp.W2"]))
(y,) = net_w([P, Lw])
(yV,) = net_w([g(V * P * g.adj(V)), [g(V * L_k * g.adj(V)) for L_k in Lw]])
eps = g.norm2(yV - g(V * y * g.adj(V))) / g.norm2(y)
g.message(f"loop words: covariance {eps}")
assert eps < 1e-26
leaves = [rad.node(w) for w in net_w.parameters()]
x = rad.node(P)
l = [rad.node(L_k) for L_k in Lw]
(yn,) = net_w([x, l], leaves)
loss = g.norm2(yn - rad.node(T, with_gradient=False)).functional(x, *l, *leaves)
fields = [P] + Lw + list(net_w.parameters())
eps = abs(loss(fields) - g.norm2(y - T)) / g.norm2(y - T)
g.message(f"loop words: node vs plain {eps}")
assert eps < 1e-14
loss.assert_gradient_error(rng, fields, fields, 1e-5, 1e-7)

# as the loop function of the transport above
params = [rho] + list(net_w.parameters())
pt = g.qcd.gauge.smear.directional_parallel_transport(
    U,
    description,
    0,
    g(even + odd),
    odd,
    params,
    loop_function=lambda sm, xp, L: net_w([sm, [L[0], g(0.5 * (L[1] + g.adj(L[1])))]], xp[1:])[0],
    loops=loops,
)
fields = g.ml.fields(U, [rho], net_w.parameters())
pt.action_log_det_jacobian().assert_gradient_error(rng, fields, fields, 1e-4, 1e-7)
Up = pt(fields)
Uinv = pt.inv(Up[0:4] + params)
eps2 = g.norm2(Uinv[0] - U[0]) / g.norm2(U[0])
g.message(f"loop words: transport inverse {eps2}")
assert eps2 < 1e-25
