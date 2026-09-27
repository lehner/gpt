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
import gpt.default as default
import gpt as g
import numpy as np


c = {}

fingerprint = g.default.get_int("--fingerprint", 0) > 1


def cayley_hamilton_function_and_gradient_3(iQ, gradient_prime, c):
    # For now use Cayley Hamilton Decomposition for traceless Hermitian 3x3 matrices,
    # see https://arxiv.org/pdf/hep-lat/0311018.pdf

    I = g.identity(iQ)

    Q = g(-1j * iQ)
    Q2 = g(Q * Q)
    Q3 = g(Q * Q2)

    c0 = g(g.trace(Q3) * (1.0 / 3.0))
    c1 = g(g.trace(Q2) * (1.0 / 2.0))

    one = g.identity(c0)

    c0max = g(2.0 * g.component.pow(1.5)(c1 / 3.0))

    theta = g.component.acos(c0 * g.component.inv(c0max))
    u = g(g.component.sqrt(c1 / 3.0) * g.component.cos(theta / 3.0))
    w = g(g.component.sqrt(c1) * g.component.sin(theta / 3.0))
    u2 = g(u * u)
    w2 = g(w * w)
    fden = g.component.inv(9.0 * u2 - w2)
    fden2 = g(fden * fden / 2.0)

    xi0 = g(g.component.sin(w) * g.component.inv(w))
    xi1 = g(g.component.cos(w) * g.component.inv(w2) - g.component.sin(w) * g.component.inv(w * w2))
    cosw = g.component.cos(w)

    emiu = g(g.component.cos(u) - 1j * g.component.sin(u))
    e2iu = g(g.component.cos(2.0 * u) + 1j * g.component.sin(2.0 * u))

    if fingerprint:
        l = g.fingerprint.log()
        l("cosw", cosw)
        l("u", u)
        l("w", w)
        l("c0max", c0max)
        l("theta", theta)
        l("fden", fden)
        l("xi0", xi0)
        l("xi1", xi1)
        l()

    # can do in stencil:
    with c.code() as cc:
        ixi0 = cc(1j * xi0)
        h0 = cc(e2iu * (u2 - w2) + emiu * ((8.0 * u2 * cosw) + (2.0 * u * (3.0 * u2 + w2) * ixi0)))
        h1 = cc(e2iu * (2.0 * u) - emiu * ((2.0 * u * cosw) - (3.0 * u2 - w2) * ixi0))
        h2 = cc(e2iu - emiu * (cosw + (3.0 * u) * ixi0))

        f0 = cc(h0 * fden)
        f1 = cc(h1 * fden)
        f2 = cc(h2 * fden)

        r01 = cc(
            (2.0 * u + 1j * 2.0 * (u2 - w2)) * e2iu
            + emiu
            * (
                (16.0 * u * cosw + 2.0 * u * (3.0 * u2 + w2) * xi0)
                + 1j * (-8.0 * u2 * cosw + 2.0 * (9.0 * u2 + w2) * xi0)
            )
        )

        r11 = cc(
            (2.0 * one + 4j * u) * e2iu
            + emiu
            * ((-2.0 * cosw + (3.0 * u2 - w2) * xi0) + 1j * ((2.0 * u * cosw + 6.0 * u * xi0)))
        )

        r21 = cc(2j * e2iu + emiu * (-3.0 * u * xi0 + 1j * (cosw - 3.0 * xi0)))

        r02 = cc(
            -2.0 * e2iu + emiu * (-8.0 * u2 * xi0 + 1j * (2.0 * u * (cosw + xi0 + 3.0 * u2 * xi1)))
        )

        r12 = cc(emiu * (2.0 * u * xi0 + 1j * (-cosw - xi0 + 3.0 * u2 * xi1)))

        r22 = cc(emiu * (xi0 - 1j * (3.0 * u * xi1)))

        b10 = cc(2.0 * u * r01 + (3.0 * u2 - w2) * r02 - (30.0 * u2 + 2.0 * w2) * f0)
        b11 = cc(2.0 * u * r11 + (3.0 * u2 - w2) * r12 - (30.0 * u2 + 2.0 * w2) * f1)
        b12 = cc(2.0 * u * r21 + (3.0 * u2 - w2) * r22 - (30.0 * u2 + 2.0 * w2) * f2)

        b20 = cc(r01 - (3.0 * u) * r02 - (24.0 * u) * f0)
        b21 = cc(r11 - (3.0 * u) * r12 - (24.0 * u) * f1)
        b22 = cc(r21 - (3.0 * u) * r22 - (24.0 * u) * f2)

        b10 *= fden2
        b11 *= fden2
        b12 *= fden2
        b20 *= fden2
        b21 *= fden2
        b22 *= fden2

    c.execute()

    if fingerprint:
        l = g.fingerprint.log()
        l("b10", b10)
        l("b11", b11)
        l("b12", b12)
        l("b20", b20)
        l("b21", b21)
        l("b22", b22)
        l()

    # assemble results
    B1 = g(b10 * I + b11 * Q + b12 * Q2)
    B2 = g(b20 * I + b21 * Q + b22 * Q2)

    U_Sigma_prime = gradient_prime

    exp_iQ = g(f0 * I + f1 * Q + f2 * Q2)

    Gamma = g(
        g.trace(U_Sigma_prime * B1) * Q
        + g.trace(U_Sigma_prime * B2) * Q2
        + f1 * U_Sigma_prime
        + f2 * Q * U_Sigma_prime
        + f2 * U_Sigma_prime * Q
    )

    Lambda = g.qcd.gauge.project.traceless_hermitian(Gamma)

    return exp_iQ, Lambda


def cayley_hamilton_function_and_gradient(x, dx, c):
    if x.otype.shape[0] == 3:
        return cayley_hamilton_function_and_gradient_3(x, dx, c)

    raise NotImplementedError()


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


def _generate(k, n, s, q):
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
    md.emit("out", [(1.0, [P[2**k - 1]])])
    return md


def _code(k, n, s):
    # Paterson-Stockmeyer block size, see above
    best = min((_generate(k, n, s, q) for q in range(1, n + 1)), key=lambda md: len(md.code) + md.n_temps)
    # field layout: output, temporaries, identity, X, H_1..H_k
    index = {"out": 0, "I": 1 + best.n_temps, "X": 2 + best.n_temps}
    for i in range(k):
        index[("H", i)] = 3 + best.n_temps + i

    def f(sym):
        return 1 + sym[1] if isinstance(sym, tuple) and sym[0] == "t" else index[sym]

    code = [(f(t), -1 if a == -1 else f(t), w, [(f(x), 0, 0) for x in fl]) for (t, a, w, fl) in best.code]
    return code, best.n_temps, best.n_products


def derivative(x, h):
    # D_k(x; h_1..h_k) for plain lattices (k = len(h); k = 0 is exp(x))
    x = g(x)
    h = [g(y) for y in h]
    k = len(h)
    n = order_base + k
    nrm = max_site_norm(x)
    s = 0 if nrm <= theta else int(np.ceil(np.log2(nrm / theta)))

    tag = f"{x.otype.__name__}_{x.grid}_{k}_{n}_{s}"
    if tag not in _kernels:
        code, n_temps, n_products = _code(k, n, s)
        _kernels[tag] = (g.local_stencil.matrix(x, [(0,) * x.grid.nd], code), n_temps)
    kernel, n_temps = _kernels[tag]

    out = g.lattice(x)
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
    kernel(out, *temps, g.identity(x), x, *hx)
    return out


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
    global c

    if x.grid.precision != g.double:
        x_dp = g.convert(x, g.double)
        dx_dp = g.convert(dx, g.double)
        A, B = function_and_gradient(x_dp, dx_dp)
        return g.convert(A, x.grid.precision), g.convert(B, x.grid.precision)

    key = f"{x.otype.__name__};{x.grid}"
    if key not in c:
        c[key] = g.compiler()
    return cayley_hamilton_function_and_gradient(x, dx, c[key])
