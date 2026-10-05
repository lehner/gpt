#!/usr/bin/env python3
#
# Node operations between arrays, site-constant fields and components
# (g.ad.reverse.broadcast_array / sum_to_array / component / embed): a layer
# y = sin(W h + b) on matrix fields (h in the first column, so that all
# products and their backward are matrix products), h assembled from scalar
# fields and outputs read as components; first derivatives and a mixed second
# derivative
#
import gpt as g
import numpy as np

rad = g.ad.reverse
rng = g.random("test")
grid = g.grid([4, 4, 4, 8], g.double)
n, n_in, n_out = 8, 3, 2
M = g.mcomplex(grid, n)
scalar = g.lattice(grid, g.ot_singlet())
st = (0, 1, lambda: g.ot_singlet())

# embed and component against direct indexing, for several indices of vector
# and matrix fields (including a matrix field decomposed into blocks)
x = g.lattice(scalar)
rng.cnormal(x)
for F in [g.mcomplex(grid, 4), g.mcomplex(grid, 16), g.vcomplex(grid, 8)]:
    shape = F.otype.shape
    for idx in [(0, 0), (1, 0), (2, 3)] if len(shape) == 2 else [(0,), (5,)]:
        e = rad.embed(x, idx, F)
        a = e[:].reshape((-1,) + shape)
        assert np.allclose(a[(slice(None),) + idx], x[:].reshape(-1))
        assert abs(g.norm2(e) / g.norm2(x) - 1) < 1e-14
        assert g.norm2(rad.component(e, idx, F) - x) == 0.0
g.message("components: embed and component against direct indexing ok")

s0 = [g.lattice(scalar) for _ in range(n_in)]
targets = [g.lattice(scalar) for _ in range(n_out)]
rng.cnormal(s0 + targets, sigma=0.5)
W0 = np.array([[rng.normal().real for _ in range(n)] for _ in range(n)], dtype=np.complex128) / n
# (the bias as the first column of a matrix)
b0 = np.zeros((n, n), dtype=np.complex128)
b0[:, 0] = [rng.normal().real for _ in range(n)]


def S(s, W, b):
    # sum_k |y_k0 - t_k|^2 with y = sin(W h + b), h = sum_j s_j e_j0
    h = sum(rad.embed(sj, (j, 0), M) for j, sj in enumerate(s))
    y = g.component.sin(rad.broadcast_array(W, M) * h + rad.broadcast_array(b, M))
    return sum(g.norm2(rad.component(y, (k, 0), M) - t) for k, t in enumerate(targets))


# plain evaluation
zero = g.lattice(scalar)
zero[:] = 0
h = g.lattice(M)
ha = np.zeros((len(s0[0][:]), n, n), dtype=np.complex128)
for i in range(n_in):
    ha[:, i, 0] = s0[i][:].reshape(-1)
h[:] = ha
Wf, bf = g.lattice(M), g.lattice(M)
Wf[:] = W0
bf[:] = b0
y = g.component.sin(g(Wf * h + bf))
ya = y[:].reshape(-1, n, n)
ref = sum(np.sum(np.abs(ya[:, k, 0] - t[:].reshape(-1)) ** 2) for k, t in enumerate(targets))
leaves = [rad.node(x) for x in s0] + [rad.node(W0), rad.node(b0)]
f = S(leaves[:n_in], leaves[n_in], leaves[n_in + 1]).functional(*leaves)
fields = s0 + [W0, b0]
eps = abs(f(fields) - ref) / ref
g.message(f"components: node vs plain value {eps}")
assert eps < 1e-14

# first derivatives w.r.t. the arrays (the scalar fields are of type
# ot_singlet, not a group type; their gradient enters the mixed second
# derivative below, contracted with a direction)
f.assert_gradient_error(rng, fields, [W0, b0], 1e-5, 1e-8)

# a mixed second derivative: F(W) = <d, grad_s S(s0, W)> (the s-gradient a
# graph in W from a nested pass), its W-gradient through the backward of the
# backward of broadcast_array, component and embed
d = [g.lattice(scalar) for _ in range(n_in)]
rng.cnormal(d)


def F_node(W1):
    s2 = [rad.node(rad.node(x)) for x in s0]
    Wc = rad.node(W1, with_gradient=False)
    bc = rad.node(rad.node(b0, with_gradient=False), with_gradient=False)
    S(s2, Wc, bc)()
    return sum(
        g.inner_product(rad.node(dj, with_gradient=False), x.gradient) for dj, x in zip(d, s2)
    )


def F_plain(W):
    s1 = [rad.node(x) for x in s0]
    S(s1, rad.node(W, with_gradient=False), rad.node(b0, with_gradient=False))()
    return sum(g.inner_product(dj, x.gradient) for dj, x in zip(d, s1)).real


W1 = rad.node(W0)
Fn = F_node(W1)
Fv = complex(rad.util.resolve(Fn())).real
eps = abs(Fv - F_plain(W0)) / abs(F_plain(W0))
g.message(f"components: nested value vs plain {eps}")
assert eps < 1e-12
dW = np.array([[rng.normal().real for _ in range(n)] for _ in range(n)], dtype=np.complex128)
analytic = np.vdot(dW, rad.util.resolve(W1.gradient)).real
e = 1e-4
numeric = (
    -F_plain(W0 + 2 * e * dW)
    + 8 * F_plain(W0 + e * dW)
    - 8 * F_plain(W0 - e * dW)
    + F_plain(W0 - 2 * e * dW)
) / (12 * e)
eps = abs(analytic - numeric) / abs(numeric)
g.message(f"components: mixed second derivative {analytic} vs {numeric}: {eps}")
assert eps < 1e-8

# componentwise functions and products of matrix fields (their backward is
# componentwise, not a matrix product)
n = 4
M4 = g.mcomplex(grid, n)
W4 = np.array([[rng.normal().real + 0.3j * rng.normal().real for _ in range(n)] for _ in range(n)])
cases = {
    "sin": lambda W: g.component.sin(rad.broadcast_array(W, M4)),
    "cos": lambda W: g.component.cos(rad.broadcast_array(W, M4)),
    "multiply": lambda W: g.component.multiply(
        rad.broadcast_array(W, M4), rad.broadcast_array(W, M4)
    ),
    "conj": lambda W: rad.transform.conj(rad.broadcast_array(W, M4)) * rad.broadcast_array(W, M4),
}
for name, op in cases.items():
    w = rad.node(W4)
    g.message(f"components: matrix {name}")
    g.norm2(op(w)).functional(w).assert_gradient_error(rng, [W4], [W4], 1e-5, 1e-7)
