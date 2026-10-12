#!/usr/bin/env python3
#
# Site-constant linear maps on lists of scalar fields (g.ad.reverse.stack /
# matrix_vector / outer_sum / dagger, gemm kernels on packed lists): values,
# first derivatives, a mixed second derivative through a two-layer network,
# and the recorded computation against numpy reference kernels
#
import gpt as g
import numpy as np
import gpt.ad.reverse.linear as linear

rad = g.ad.reverse
rng = g.random("test")
grid = g.grid([4, 4, 4, 8], g.double)


def fields(n, sigma=1.0):
    x = [g.complex(grid) for _ in range(n)]
    rng.cnormal(x, sigma=sigma)
    return x


def matrix(m, n, scale=1.0):
    return scale * np.array(
        [[complex(rng.normal().real, rng.normal().real) for _ in range(n)] for _ in range(m)]
    )


def columns(x):
    return np.stack([f[:].reshape(-1) for f in x], axis=1)


# values
for m, n in [(5, 3), (3, 5), (16, 16)]:
    h, a, W = fields(n), fields(m), matrix(m, n)
    assert np.allclose(columns(rad.matrix_vector(W, h)), columns(h) @ W.T)
    # (columns holds the local sites of this rank: summed over the ranks)
    ref = grid.globalsum(np.ascontiguousarray(columns(a).T @ np.conj(columns(h))))
    assert np.allclose(rad.outer_sum(a, h), ref)
    assert np.allclose(rad.dagger(W), np.conj(W).T)
g.message("linear: values ok")

# first derivatives: |W h - t|^2 w.r.t. W and h, |outer_sum(a, b) - T|^2 w.r.t. a and b
m, n = 4, 3
h, t, W = fields(n), fields(m), matrix(m, n)
lW, lh = rad.node(W), [rad.node(x) for x in h]
y = rad.matrix_vector(lW, lh)
L = sum(g.norm2(y[i] - rad.node(t[i], with_gradient=False)) for i in range(m))
L.functional(lW, *lh).assert_gradient_error(rng, [W] + h, [W] + h, 1e-5, 1e-8)
a, b, T = fields(m), fields(n), matrix(m, n)
la, lb = [rad.node(x) for x in a], [rad.node(x) for x in b]
D = rad.outer_sum(la, lb)
# (a functional of the array through a contraction with fields)
L = sum(
    g.norm2(rad.matrix_vector(rad.dagger(D), [rad.node(x, with_gradient=False) for x in t])[j])
    for j in range(n)
)
L.functional(*la, *lb).assert_gradient_error(rng, a + b, a + b, 1e-5, 1e-8)

# the plain backward of matrix_vector: flows only into the entries of h that
# carry a gradient (here not the constant last entry, e.g. a bias field), and
# a second pass over retained values (without the packed h of the forward)
# gives the same gradients as the first
m, n = 4, 3
h, t, W = fields(n), fields(m), matrix(m, n)
lW, lh = rad.node(W), [rad.node(x) for x in h[:-1]] + [rad.node(h[-1], with_gradient=False)]
y = rad.matrix_vector(lW, lh)
L = sum(g.norm2(y[i] - rad.node(t[i], with_gradient=False)) for i in range(m))
L(retain_values=True)
first = [lW.gradient] + [x.gradient for x in lh[:-1]]
assert lh[-1].gradient is None
L.backward(retain_values=True)
second = [lW.gradient] + [x.gradient for x in lh[:-1]]
assert np.array_equal(first[0], second[0])
for x, z in zip(first[1:], second[1:]):
    assert g.norm2(x - z) == 0.0
# and they agree with the gradients of the pass that differentiates every entry
lh_all = [rad.node(x) for x in h]
lW_all = rad.node(W)
y = rad.matrix_vector(lW_all, lh_all)
sum(g.norm2(y[i] - rad.node(t[i], with_gradient=False)) for i in range(m)).backward()
assert np.array_equal(first[0], lW_all.gradient)
for x, z in zip(first[1:], lh_all[:-1]):
    assert g.norm2(x - z.gradient) == 0.0
g.message("linear: plain backward (flowed entries, without residual) ok")


# a two-layer network S(h, W1, W2) = sum_k |sin(W2 sin(W1 h))_k - t_k|^2 and the
# mixed second derivative F(W1) = <d, grad_h S>, d/dW1 F through the backward
# of the backward of matrix_vector and outer_sum
n_in, n_hidden, n_out = 3, 4, 2
h0 = fields(n_in, 0.5)
targets = [rad.node(x, with_gradient=False) for x in fields(n_out)]
W1, W2 = matrix(n_hidden, n_in, 0.5), matrix(n_out, n_hidden, 0.5)
d = fields(n_in)


def S_(h, W1, W2):
    y1 = rad.matrix_vector(W1, h)
    y1 = [g.component.sin(y1[i]) for i in range(n_hidden)]
    y2 = rad.matrix_vector(W2, y1)
    return sum(g.norm2(g.component.sin(y2[k]) - t) for k, t in enumerate(targets))


def F_plain(W):
    h1 = [rad.node(x) for x in h0]
    S_(h1, rad.node(W, with_gradient=False), rad.node(W2, with_gradient=False))()
    return sum(g.inner_product(dj, x.gradient) for dj, x in zip(d, h1)).real


def F_recorded():
    # the reverse pass of S is recorded (create_graph=True, restricted to the
    # h leaves): h.gradient is a node graph over the same leaves, W1 among
    # them, so the backward of <d, grad_h S> deposits d/dW1 F into lW1
    lW1 = rad.node(W1)
    lh = [rad.node(x) for x in h0]
    S_(lh, lW1, rad.node(W2, with_gradient=False)).backward(create_graph=True, wrt=lh)
    F = sum(g.inner_product(dj, x.gradient) for dj, x in zip(d, lh))
    value = complex(F()).real
    return value, lW1.gradient


value, gradient = F_recorded()
eps = abs(value - F_plain(W1)) / abs(F_plain(W1))
g.message(f"linear: recorded value vs plain {eps}")
assert eps < 1e-12
dW = matrix(n_hidden, n_in)
analytic = np.vdot(dW, gradient).real
e = 1e-4
numeric = (
    -F_plain(W1 + 2 * e * dW)
    + 8 * F_plain(W1 + e * dW)
    - 8 * F_plain(W1 - e * dW)
    + F_plain(W1 - 2 * e * dW)
) / (12 * e)
eps = abs(analytic - numeric) / abs(numeric)
g.message(f"linear: mixed second derivative {analytic} vs {numeric}: {eps}")
assert eps < 1e-8 and np.all(np.isfinite(gradient))

# the recorded computation with a numpy reference gemm (the packed buffers read
# on the host; the packing itself is checked by the values above) gives the
# same value and gradient as the gemm kernels
kernel = linear._gemm


def gemm_reference(a, b, shape_c, op_a="N", op_b="N"):
    A, B = np.asarray(a.to_array())[0], np.asarray(b.to_array())[0]
    A = np.conj(A).T if op_a == "H" else A
    B = np.conj(B).T if op_b == "H" else B
    C = A @ B
    assert C.shape == shape_c
    return g.accelerator.buffer(np.ascontiguousarray(C).reshape((1,) + shape_c))


linear._gemm = gemm_reference
value_ref, gradient_ref = F_recorded()
linear._gemm = kernel
eps = max(
    abs(value - value_ref) / abs(value_ref),
    np.max(np.abs(gradient - gradient_ref)) / np.max(np.abs(gradient_ref)),
)
g.message(f"linear: gemm kernels vs numpy reference in the recorded computation {eps}")
assert eps < 1e-12

# the zero of an array (or tensor) gradient is assigned, not multiplied: a
# representative of the container may be uninitialized memory (here: memory
# that held nan before), and 0 times nan is nan
from gpt.ad.reverse.util import container

for shape in [(16, 16), (12,), (5, 3)]:
    c = container(np.ndarray, shape, np.complex128)
    for _ in range(50):
        junk = np.full(shape, np.nan, dtype=np.complex128)
        del junk
        z = c.zero()
        assert np.all(z == 0), "zero of an array container is not zero"
g.message("linear: array zeros ok")
