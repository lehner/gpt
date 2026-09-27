#
#    GPT - Grid Python Toolkit
#    Copyright (C) 2023-2026  Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)
#
#    This program is free software; you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation; either version 2 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License along
#    with this program; if not, write to the Free Software Foundation, Inc.,
#    51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
#
# Differentiable matrix exponential as a self-similar tower of fused kernels.
#
# The node exp(X) is the k=0 member of the family
#
#     D_k(X; H_1, ..., H_k) = d^k exp_X (H_1, ..., H_k),
#
# the k-th directional derivative (symmetric, multilinear in the H_i).  Every
# D_k is holomorphic in its arguments, and the reverse-mode flows of D_k are
# again members of the family (W = the flow into D_k):
#
#     flow into X   :  D_{k+1}(X^dag; H_1^dag, ..., H_k^dag, W)
#     flow into H_i :  D_k(X^dag; H_1^dag, ..., W, ..., H_k^dag)   (W at slot i)
#
# i.e., the adjoint of a derivative of exp is a derivative of exp at X^dag.
# (For k=0 this is the familiar int_0^1 e^{s X^dag} W e^{(1-s) X^dag} ds.)
# So, as for stencils, the gradient of exp is exp: a nested pass builds D_k
# nodes one level down, and at the plain level each D_k is ONE compiled local
# kernel instead of a graph of O(100) elementwise node operations.
#
# Plain evaluation: exp of the multi-dual number X + sum_i eps_i H_i with
# eps_i^2 = 0, whose eps_1...eps_k coefficient is D_k.  A component is kept
# per subset S of {1..k}; scaling and squaring on the multi-dual argument
# (exp(Y) = exp(Y / 2^s)^(2^s)), with a Horner-evaluated Taylor polynomial.
#
import gpt as g
import numpy as np
from gpt.ad.reverse.util import is_node, nodify, value_of


# after scaling, the largest site norm is at most theta; the Taylor order is
# chosen such that the truncation error of the k-th derivative component,
# theta^(n+1-k) / (n+1-k)!, is below 1e-17
theta = 0.25
order_base = 12

_kernels = {}


def _max_site_norm(x):
    # upper bound on the largest site norm |x|_F, computed on the device:
    # (sum_x |x|_F^16)^(1/16) >= max_x |x|_F, and overestimates it by at most
    # V^(1/16) (a factor 2 at 16^4), i.e., by at most one extra squaring
    n2 = g(g.trace(g.adj(x) * x))
    y = n2
    for _ in range(3):
        y = g(y * y)
    bound = g.sum(y).real ** (1.0 / 16.0)
    if np.isfinite(bound):
        return bound
    # |x|^16 overflowed (huge arguments in single precision): exact maximum
    local = float(np.max(n2[:].real)) if n2.grid.gsites > 0 else 0.0
    grid = x.grid
    per_rank = np.zeros(grid.Nprocessors, dtype=np.float64)
    per_rank[grid.processor] = local
    grid.globalsum(per_rank)
    return float(np.max(per_rank)) ** 0.5


def _code(k, n, s):
    # field layout: 0 = output, then two banks of 2^k components, then the
    # identity, X, H_1..H_k
    nc = 2**k
    bank = [[1 + i for i in range(nc)], [1 + nc + i for i in range(nc)]]
    f_I = 1 + 2 * nc
    f_X = f_I + 1
    f_H = [f_X + 1 + i for i in range(k)]
    scale = 1.0 / 2**s

    code = []

    def emit(target, terms):
        # target = sum of weight * product(factors); first write is fresh
        first = True
        for w, factors in terms:
            code.append((target, -1 if first else target, w, [(f, 0, 0) for f in factors]))
            first = False

    # p = I + x / n  (x = scaled multi-dual argument)
    cur = 0
    w = scale / n
    nonzero = {0}
    emit(bank[cur][0], [(1.0, [f_I]), (w, [f_X])])
    for i in range(k):
        emit(bank[cur][1 << i], [(w, [f_H[i]])])
        nonzero.add(1 << i)

    # Horner: p = I + (x / j) p
    for j in range(n - 1, 0, -1):
        w = scale / j
        nxt = 1 - cur
        new_nonzero = set()
        for S in range(nc):
            terms = [(1.0, [f_I])] if S == 0 else []
            if S in nonzero:
                terms.append((w, [f_X, bank[cur][S]]))
            for i in range(k):
                if S & (1 << i) and (S ^ (1 << i)) in nonzero:
                    terms.append((w, [f_H[i], bank[cur][S ^ (1 << i)]]))
            if terms:
                emit(bank[nxt][S], terms)
                new_nonzero.add(S)
        nonzero = new_nonzero
        cur = nxt

    # squaring: p[S] = sum_{T subset S} p[T] p[S \ T]
    for _ in range(s):
        nxt = 1 - cur
        new_nonzero = set()
        for S in range(nc):
            terms = []
            T = S
            while True:
                if T in nonzero and (S ^ T) in nonzero:
                    terms.append((1.0, [bank[cur][T], bank[cur][S ^ T]]))
                if T == 0:
                    break
                T = (T - 1) & S
            if terms:
                emit(bank[nxt][S], terms)
                new_nonzero.add(S)
        nonzero = new_nonzero
        cur = nxt

    top = nc - 1
    if top in nonzero:
        emit(0, [(1.0, [bank[cur][top]])])
    else:
        # cannot happen for the multilinear coefficient, kept for safety
        emit(0, [(0.0, [f_I])])
    return code, 1 + 2 * nc + 1 + 1 + k


def _plain(x, h):
    x = g(x)
    h = [g(y) for y in h]
    k = len(h)
    n = order_base + k
    nrm = _max_site_norm(x)
    s = 0 if nrm <= theta else int(np.ceil(np.log2(nrm / theta)))

    tag = f"{x.otype.__name__}_{x.grid}_{k}_{n}_{s}"
    if tag not in _kernels:
        code, n_fields = _code(k, n, s)
        _kernels[tag] = (g.local_stencil.matrix(x, [(0,) * x.grid.nd], code), n_fields)
    kernel, n_fields = _kernels[tag]

    out = g.lattice(x)
    temps = [g.lattice(x) for _ in range(2 * 2**k)]
    one = g.identity(x)
    # the kernel runs on the matrix storage; directions of a different
    # (e.g. algebra) otype are re-labelled to X's otype
    hx = []
    for y in h:
        if y.otype.__name__ != x.otype.__name__:
            z = g.lattice(x)
            z @= y
            y = z
        hx.append(y)
    fields = [out] + temps + [one, x] + hx
    assert len(fields) == n_fields
    kernel(*fields)
    return out


def derivative(x, h):
    # D_k(x; h_1..h_k); plain values run the fused kernel, node values build a
    # node whose backward is again a D (one level down)
    args = [x] + list(h)
    if not any(is_node(a) for a in args):
        return _plain(x, h)

    args = nodify(*args) if len(args) > 1 else (args[0],)
    xn, hn = args[0], list(args[1:])
    k = len(hn)

    def _adj(v):
        return g.adj(v)

    def _backward_x(z):
        return (1, derivative(_adj(value_of(xn)), [_adj(value_of(c)) for c in hn] + [z.gradient]))

    def _backward_h(i):
        def _b(z):
            return (
                1,
                derivative(
                    _adj(value_of(xn)),
                    [z.gradient if j == i else _adj(value_of(c)) for j, c in enumerate(hn)],
                ),
            )

        return _b

    return g.ad.reverse.node_op(
        tuple(args),
        lambda: derivative(value_of(xn), [value_of(c) for c in hn]),
        (_backward_x,) + tuple(_backward_h(i) for i in range(k)),
        xn._container,
        f"exp_d{k}",
    )


def _taylor_graph(x):
    # generic node-op fallback for non-lattice (tensor / scalar) values
    fac = 1.0
    base = 128.0
    nbase = 7
    x = x / base
    c = x
    r = g.identity(x)
    for i in range(1, 10):
        fac /= float(i)
        r = r + fac * c
        c = c * x
    for i in range(nbase):
        r = r * r
    return r


# gives 1e-14 / 1e-15 errors up to at least |x| < 10


def function(x):
    if x._container.tag[0] != g.lattice:
        return _taylor_graph(x)
    return derivative(x, [])


def function_and_gradient(x, dx):
    assert False
