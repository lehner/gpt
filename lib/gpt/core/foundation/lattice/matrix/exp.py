#
#    GPT - Grid Python Toolkit
#    Copyright (C) 2020-22  Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)
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
import gpt as g
import numpy as np


# Matrix exponential and its derivatives D_k(X; H_1..H_k) = d^k exp_X(H_1..H_k),
# each as ONE compiled local kernel: exp of the multi-dual number
# X + sum_i eps_i H_i (eps_i^2 = 0), whose eps_1...eps_k coefficient is D_k.
# A component is kept per subset S of {1..k} (k = 0 is exp itself).
#
# Scaling and squaring, exp(Y) = exp(Y / 2^s)^(2^s), with the scale folded
# into the Taylor coefficients, and the Taylor polynomial evaluated by
# Paterson-Stockmeyer: p = sum_b C_b y^b with y = x^q and C_b a linear
# combination of I, x, ..., x^(q-1), so that a degree-n polynomial takes
# about 2 sqrt(n) multi-dual products instead of n.  q is chosen per (k, n)
# from the generated code: the local-stencil temporaries are full lattices,
# so their memory traffic costs about as much as a code entry, and q
# minimizes (code entries + temporaries) rather than the number of products.
#
# Accuracy: after scaling the largest site norm is at most theta and the
# Taylor order is such that the truncation error of the k-th derivative,
# theta^(n+1-k) / (n+1-k)!, is below 1e-17 (relative).

theta = 0.25
order_base = 12

_kernels = {}


def max_site_norm(x):
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


class _multi_dual_code:
    # symbolic multi-dual algebra emitting local-stencil code; a multi-dual
    # value is a dict {subset mask: field}, absent components are zero
    def __init__(self, k):
        self.k = k
        self.code = []
        self.n_temps = 0
        self.n_products = 0

    def temp(self):
        self.n_temps += 1
        return ("t", self.n_temps - 1)

    def emit(self, target, terms):
        # target = sum of weight * product(factors); the first write is fresh
        for i, (w, factors) in enumerate(terms):
            self.code.append((target, -1 if i == 0 else target, w, factors))
            self.n_products += len(factors) - 1

    def combine(self, targets, lincomb, product):
        # result[S] = sum_i c_i A_i[S] (+ c_I at S = 0) + sum_T P[T] Y[S \ T]
        result = {}
        for S in range(2**self.k):
            terms = []
            if S == 0 and lincomb[0] != 0.0:
                terms.append((lincomb[0], ["I"]))
            for c, A in lincomb[1]:
                if S in A and c != 0.0:
                    terms.append((c, [A[S]]))
            if product is not None:
                P, Y = product
                T = S
                while True:
                    if T in P and (S ^ T) in Y:
                        terms.append((1.0, [P[T], Y[S ^ T]]))
                    if T == 0:
                        break
                    T = (T - 1) & S
            if terms:
                result[S] = targets(S)
                self.emit(result[S], terms)
        return result

    def mul(self, A, B, targets=None):
        return self.combine(targets or (lambda S: self.temp()), (0.0, []), (A, B))


def _generate(k, n, s, q, outputs):
    md = _multi_dual_code(k)
    x = {0: "X"}
    for i in range(k):
        x[1 << i] = ("H", i)
    # coefficients of the scaled Taylor polynomial, c_j = 2^(-s j) / j!
    c = [1.0]
    for j in range(1, n + 1):
        c.append(c[-1] / j / 2**s)
    powers = [None, x]
    for _ in range(2, q + 1):
        powers.append(md.mul(powers[-1], x))
    y = powers[q]
    banks = [{S: md.temp() for S in range(2**k)} for _ in range(2)]

    def block(b):
        # C_b = sum_{i < q} c_{bq+i} x^i  as (coefficient of I, [(c, x^i)])
        cI = c[b * q]
        terms = [(c[b * q + i], powers[i]) for i in range(1, q) if b * q + i <= n]
        return cI, terms

    B = n // q
    cur = 0
    P = md.combine(lambda S: banks[cur][S], block(B), None)
    for b in range(B - 1, -1, -1):
        cur = 1 - cur
        P = md.combine(lambda S: banks[cur][S], block(b), (P, y))
    for _ in range(s):
        cur = 1 - cur
        P = md.mul(P, P, lambda S: banks[cur][S])
    for i, S in enumerate(outputs):
        md.emit(("out", i), [(1.0, [P[S]])])
    return md


def _code(k, n, s, outputs):
    # Paterson-Stockmeyer block size, see above
    best = min(
        (_generate(k, n, s, q, outputs) for q in range(1, n + 1)),
        key=lambda md: len(md.code) + md.n_temps,
    )
    # field layout: outputs, temporaries, identity, X, H_1..H_k
    m = len(outputs)
    index = {"I": m + best.n_temps, "X": m + 1 + best.n_temps}
    for i in range(k):
        index[("H", i)] = m + 2 + best.n_temps + i

    def f(sym):
        if isinstance(sym, tuple) and sym[0] == "t":
            return m + sym[1]
        if isinstance(sym, tuple) and sym[0] == "out":
            return sym[1]
        return index[sym]

    code = [(f(t), -1 if a == -1 else f(t), w, [(f(x), 0, 0) for x in fl]) for (t, a, w, fl) in best.code]
    return code, best.n_temps, best.n_products


def _evaluate(x, h, outputs):
    # the multi-dual components `outputs` (subset masks) of exp(x + sum eps_i h_i)
    x = g(x)
    h = [g(y) for y in h]
    k = len(h)
    n = order_base + k
    nrm = max_site_norm(x)
    s = 0 if nrm <= theta else int(np.ceil(np.log2(nrm / theta)))

    tag = f"{x.otype.__name__}_{x.grid}_{k}_{n}_{s}_{outputs}"
    if tag not in _kernels:
        code, n_temps, n_products = _code(k, n, s, outputs)
        m = len(outputs)
        kernel = g.local_stencil.matrix(
            x, [(0,) * x.grid.nd], code, temporaries=list(range(m, m + n_temps))
        )
        _kernels[tag] = (kernel, n_temps)
    kernel, n_temps = _kernels[tag]

    out = [g.lattice(x) for _ in outputs]
    temps = [g.lattice(x) for _ in range(n_temps)]
    # the kernel runs on the matrix storage; directions of a different
    # (e.g. algebra) otype are re-labelled to X's otype
    hx = []
    for y in h:
        if y.otype.__name__ != x.otype.__name__:
            z = g.lattice(x)
            z @= y
            y = z
        hx.append(y)
    kernel(*out, *temps, g.identity(x), x, *hx)
    return out


def derivative(x, h):
    # D_k(x; h_1..h_k) for plain lattices (k = len(h); k = 0 is exp(x))
    return _evaluate(x, h, [2 ** len(h) - 1])[0]


def function(i):
    i = g.eval(i)  # accept expressions
    x = g.convert(i, g.double) if i.grid.precision != g.double else i
    if isinstance(x, g.lattice) and len(x.v_obj) == 1:
        o = derivative(x, [])
    else:
        # types stored in several v_obj: scaled Taylor series in lattice
        # operations, with the same scaling rule and order
        nrm = max_site_norm(x)
        s = 0 if nrm <= theta else int(np.ceil(np.log2(nrm / theta)))
        xs = g(x * (1.0 / 2**s))
        o = g.identity(x)
        xn = g.copy(xs)
        o += xn
        nfac = 1.0
        for j in range(2, order_base + 1):
            nfac /= j
            xn @= xn * xs
            o += xn * nfac
        for j in range(s):
            o @= o * o
    if i.grid.precision != g.double:
        r = g.lattice(i)
        g.convert(r, o)
        o = r
    return o


def function_and_gradient(x, dx):
    # exp(x) and the stout-smearing force factor
    #   Lambda = traceless_hermitian(Gamma),  d tr(dx exp(iQ)) = tr(Gamma dQ),
    # for x = iQ (see https://arxiv.org/pdf/hep-lat/0311018.pdf); from
    # tr(dx D_1(x; H)) = tr(D_1(x; dx) H) it follows that Gamma = i D_1(x; dx)
    # up to the identity, so both come from ONE D_1 kernel (components 0 and
    # top of the multi-dual exponential), for any N
    if x.grid.precision != g.double:
        A, B = function_and_gradient(g.convert(x, g.double), g.convert(dx, g.double))
        return g.convert(A, x.grid.precision), g.convert(B, x.grid.precision)

    exp_x, d1 = _evaluate(x, [dx], [0, 1])
    return exp_x, g.qcd.gauge.project.traceless_hermitian(g(1j * d1))
