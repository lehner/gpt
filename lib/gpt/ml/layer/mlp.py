#
#    GPT - Grid Python Toolkit
#    Copyright (C) 2026  Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)
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
# Site-local layers for gauge-covariant functions of a matrix field P:
#
#   f(P) = P + sum_w c_w(I(P)) w(P)
#
# with the invariants I (matrix_invariants), coefficient functions of them
# (mlp) and covariant words w in P and P^dag (matrix_words).
#
import gpt as g
import numpy as np
from gpt.ml.function import function


def _unit(template):
    # the unit scalar field, of the type of traces of the template (the
    # scalar type of all slots below)
    one = g(g.trace(g.identity(template)))
    one[:] = 1
    return one


class matrix_invariants(function):
    """I = the standardized invariants of a matrix field P (N x N), per site:
    Re tr P / N, Im tr P / N, tr P P^dag / N, Re tr P^2 / N, Im tr P^2 / N,
    each as (q - mean) inv_std (real complex fields).  Constants: mean,
    inv_std (set by calibrate to the mean and inverse standard deviation over
    the sites of the calibration samples, 1 for an invariant that is
    constant; frozen, to keep the layer site-local).  No parameters."""

    n = 5

    def __init__(self, template):
        self.N = template.otype.shape[0]
        self.one = _unit(template)
        s = g.lattice(self.one)
        super().__init__(
            [("P", template)],
            [("I", [s] * self.n)],
            [],
            [("mean", [0.0] * self.n), ("inv_std", [1.0] * self.n)],
        )

    def initialize(self, rng, scale=None):
        pass

    def _raw(self, P):
        r = 1.0 / self.N
        t1 = g(g.trace(P) * r)
        t2 = g(g.trace(P * P) * r)
        return [
            g.component.real(t1),
            g.component.imag(t1),
            g.component.real(g(g.trace(P * g.adj(P)) * r)),
            g.component.real(t2),
            g.component.imag(t2),
        ]

    def calibrate(self, samples):
        q = [[g(x) for x in self._raw(P)] for (P,) in samples]
        mean, inv_std = [], []
        for k in range(self.n):
            n = sum(x[k].grid.gsites for x in q)
            m = sum(g.sum(x[k]).real for x in q) / n
            m2 = sum(g.sum(g(x[k] * x[k])).real for x in q) / n
            mean.append(m)
            # (a constant invariant is only shifted)
            var = m2 - m**2
            inv_std.append(1.0 / np.sqrt(var) if var > 1e-20 * (1.0 + m**2) else 1.0)
        self["mean"], self["inv_std"] = mean, inv_std

    def evaluate(self, inputs, parameters, constants):
        (P,) = inputs
        mean, inv_std = constants
        # (a sum: node subtraction requires identical containers)
        return [
            [
                g(q * inv_std[k] + (-mean[k] * inv_std[k]) * self.one)
                for k, q in enumerate(self._raw(P))
            ]
        ]


class mlp(function):
    """y = W_L sin(... sin(W_1 x + b_1) ...) + b_L on lists of complex scalar
    fields (per site; n_in inputs, n_out outputs, `depth` hidden layers of
    `width`).  The hidden weights and biases are real (used as Re w, so
    that sin sees real arguments for real inputs), the output layer is
    complex and initialized at scale (default 0: the output is zero, with
    nonzero gradients w.r.t. the output layer), the hidden layers at
    O(1) / sqrt(fan-in).  (sin: a smooth activation available as a node
    operation at any nesting depth.)"""

    def __init__(self, template, n_in, n_out, width=16, depth=2, scale=0.0):
        self.n_in, self.n_out, self.width, self.depth, self.scale = n_in, n_out, width, depth, scale
        self.one = _unit(template)
        sizes = [n_in] + [width] * depth + [n_out]
        self.sizes = sizes
        parameters = []
        for l in range(len(sizes) - 1):
            parameters += [
                (f"W{l}", [0j] * (sizes[l] * sizes[l + 1])),
                (f"b{l}", [0j] * sizes[l + 1]),
            ]
        s = g.lattice(self.one)
        super().__init__([("x", [s] * n_in)], [("y", [s] * n_out)], parameters)

    def initialize(self, rng, scale=None):
        scale = self.scale if scale is None else scale
        L = len(self.sizes) - 1
        for l in range(L):
            n0, n1 = self.sizes[l], self.sizes[l + 1]
            if l < L - 1:
                self[f"W{l}"] = [complex(rng.normal().real / np.sqrt(n0)) for _ in range(n0 * n1)]
                self[f"b{l}"] = [complex(rng.normal().real) for _ in range(n1)]
            else:
                self[f"W{l}"] = [
                    scale * rng.normal_element(0j) / np.sqrt(2 * n0) for _ in range(n0 * n1)
                ]
                self[f"b{l}"] = [scale * rng.normal_element(0j) / np.sqrt(2) for _ in range(n1)]

    def evaluate(self, inputs, parameters, constants):
        (x,) = inputs
        L = len(self.sizes) - 1
        for l in range(L):
            W, b = parameters[2 * l], parameters[2 * l + 1]
            n0, n1 = self.sizes[l], self.sizes[l + 1]
            hidden = l < L - 1
            w = (lambda p: g.component.real(p)) if hidden else (lambda p: p)
            y = []
            for j in range(n1):
                acc = w(b[j]) * self.one
                for i in range(n0):
                    acc = acc + x[i] * w(W[j * n0 + i])
                acc = g(acc)
                y.append(g.component.sin(acc) if hidden else acc)
            x = y
        return [x]


class matrix_words(function):
    """f = P + sum_w c_w w(P) for the covariant words P^2, P^dag, P P^dag,
    P^dag P, P^3, P^2 P^dag of a matrix field P and coefficient fields c
    (complex scalar fields, per site).  No parameters: the coefficients are
    inputs (e.g. the output of an mlp of the invariants of P)."""

    names = ["P^2", "P^dag", "P P^dag", "P^dag P", "P^3", "P^2 P^dag"]
    n = len(names)

    def __init__(self, template):
        s = g.lattice(_unit(template))
        super().__init__([("P", template), ("c", [s] * self.n)], [("f", template)], [])

    def initialize(self, rng, scale=None):
        pass

    def evaluate(self, inputs, parameters, constants):
        P, c = inputs
        Pd = g.adj(P)
        P2 = g(P * P)
        words = [P2, g(Pd), g(P * Pd), g(Pd * P), g(P2 * P), g(P2 * Pd)]
        f = P
        for ci, w in zip(c, words):
            f = g(f + w * ci)
        return [f]
