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
    and with n_loops > 0 of each field L_k of a second input L (a list of
    n_loops N x N fields, e.g. the fixed loops of
    directional_parallel_transport): Re tr L_k / N, Im tr L_k / N (for SU(2)
    and SU(3) loops these determine the eigenvalues; tr L L^dag / N = 1), or
    with loop_imag=False Re tr L_k / N only (e.g. for sums of loops over
    orbits of a symmetry that reverses orientations, and even under charge
    conjugation; loop_imag may also be a list, one flag per loop), and with
    mixed=True further Re tr P L_k / N, Im tr P L_k / N per loop.
    Each as (q - mean) inv_std (real complex fields).  Constants: mean,
    inv_std (set by calibrate to the mean and inverse standard deviation over
    the sites of the calibration samples, 1 for an invariant that is
    constant; frozen, to keep the layer site-local).  No parameters."""

    # invariants of P, per loop, and in total without loops (an instance
    # has its own total n)
    n_P = 5
    n_L = 2
    n = n_P

    def __init__(self, template, n_loops=0, loop_imag=True, mixed=False):
        self.N = template.otype.shape[0]
        if not isinstance(loop_imag, (list, tuple)):
            loop_imag = [loop_imag] * n_loops
        assert len(loop_imag) == n_loops
        self.n_loops, self.loop_imag, self.mixed = n_loops, list(loop_imag), mixed
        self.n = self.n_P + sum(2 if x else 1 for x in self.loop_imag) + (2 * n_loops if mixed else 0)
        self.one = _unit(template)
        s = g.lattice(self.one)
        inputs = [("P", template)]
        if n_loops > 0:
            inputs.append(("L", [template] * n_loops))
        super().__init__(
            inputs,
            [("I", [s] * self.n)],
            [],
            [("mean", [0.0] * self.n), ("inv_std", [1.0] * self.n)],
        )

    def initialize(self, rng, scale=None):
        pass

    def _raw(self, P, L=[]):
        r = 1.0 / self.N
        t1 = g(g.trace(P) * r)
        t2 = g(g.trace(P * P) * r)
        q = [
            g.component.real(t1),
            g.component.imag(t1),
            g.component.real(g(g.trace(P * g.adj(P)) * r)),
            g.component.real(t2),
            g.component.imag(t2),
        ]
        for L_k, imag in zip(L, self.loop_imag):
            t = g(g.trace(L_k) * r)
            q += [g.component.real(t)] + ([g.component.imag(t)] if imag else [])
        if self.mixed:
            for L_k in L:
                t = g(g.trace(P * L_k) * r)
                q += [g.component.real(t), g.component.imag(t)]
        return q

    def calibrate(self, samples):
        q = [[g(x) for x in self._raw(*inputs)] for inputs in samples]
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
        mean, inv_std = constants
        # (a sum: node subtraction requires identical containers)
        return [
            [
                g(q * inv_std[k] + (-mean[k] * inv_std[k]) * self.one)
                for k, q in enumerate(self._raw(*inputs))
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
    operation at any nesting depth.)

    Parameters: W<l>, an n_out_l x (n_in_l + 1) array per layer whose last
    column is the bias b_l (the input list is extended by the unit field).
    Each layer is one g.ad.reverse.matrix_vector of the list of fields (a
    gemm over the sites on the packed list), whose weight gradient is an
    outer_sum over the sites."""

    def __init__(self, template, n_in, n_out, width=16, depth=2, scale=0.0):
        self.n_in, self.n_out, self.width, self.depth, self.scale = n_in, n_out, width, depth, scale
        self.one = _unit(template)
        sizes = [n_in] + [width] * depth + [n_out]
        self.sizes = sizes
        parameters = [
            (f"W{l}", np.zeros((sizes[l + 1], sizes[l] + 1), dtype=np.complex128))
            for l in range(len(sizes) - 1)
        ]
        s = g.lattice(self.one)
        super().__init__([("x", [s] * n_in)], [("y", [s] * n_out)], parameters)

    def initialize(self, rng, scale=None):
        scale = self.scale if scale is None else scale
        L = len(self.sizes) - 1
        for l in range(L):
            n0, n1 = self.sizes[l], self.sizes[l + 1]
            if l < L - 1:
                W = [complex(rng.normal().real / np.sqrt(n0)) for _ in range(n0 * n1)]
                b = [complex(rng.normal().real) for _ in range(n1)]
            else:
                W = [scale * rng.normal_element(0j) / np.sqrt(2 * n0) for _ in range(n0 * n1)]
                b = [scale * rng.normal_element(0j) / np.sqrt(2) for _ in range(n1)]
            Wb = np.zeros((n1, n0 + 1), dtype=np.complex128)
            Wb[:, :n0] = np.array(W).reshape(n1, n0)
            Wb[:, n0] = b
            self[f"W{l}"] = Wb

    def evaluate(self, inputs, parameters, constants):
        rad = g.ad.reverse
        (x,) = inputs
        L = len(self.sizes) - 1
        for l in range(L):
            W = parameters[l]
            hidden = l < L - 1
            if hidden:
                W = g.component.real(W)
            y = rad.matrix_vector(W, list(x) + [self.one])
            x = [g.component.sin(y[j]) if hidden else y[j] for j in range(self.sizes[l + 1])]
        return [x]


class matrix_words(function):
    """f = P + sum_w c_w w(P) for the covariant words P^2, P^dag, P P^dag,
    P^dag P, P^3, P^2 P^dag of a matrix field P and coefficient fields c
    (complex scalar fields, per site).  No parameters: the coefficients are
    inputs (e.g. the output of an mlp of the invariants of P).

    With n_loops > 0 a third input L (a list of n_loops N x N fields that
    transform like P, e.g. fixed loops at x) and the words linear in L
    after those of P: per loop L_k, P L_k, L_k P, and L_k^dag if
    loop_adjoint (a flag, or a list of flags per loop; e.g. False for a
    hermitian L_k)."""

    names = ["P^2", "P^dag", "P P^dag", "P^dag P", "P^3", "P^2 P^dag"]
    n = len(names)

    def __init__(self, template, n_loops=0, loop_adjoint=True):
        if not isinstance(loop_adjoint, (list, tuple)):
            loop_adjoint = [loop_adjoint] * n_loops
        assert len(loop_adjoint) == n_loops
        self.n_loops, self.loop_adjoint = n_loops, list(loop_adjoint)
        self.names = list(matrix_words.names)
        for k, adjoint in enumerate(self.loop_adjoint):
            self.names += [f"L{k}", f"P L{k}", f"L{k} P"] + ([f"L{k}^dag"] if adjoint else [])
        self.n = len(self.names)
        s = g.lattice(_unit(template))
        inputs = [("P", template), ("c", [s] * self.n)]
        if n_loops > 0:
            inputs.append(("L", [template] * n_loops))
        super().__init__(inputs, [("f", template)], [])

    def initialize(self, rng, scale=None):
        pass

    def evaluate(self, inputs, parameters, constants):
        P, c = inputs[0:2]
        L = inputs[2] if self.n_loops > 0 else []
        Pd = g.adj(P)
        P2 = g(P * P)
        words = [P2, g(Pd), g(P * Pd), g(Pd * P), g(P2 * P), g(P2 * Pd)]
        for L_k, adjoint in zip(L, self.loop_adjoint):
            words += [L_k, g(P * L_k), g(L_k * P)] + ([g(g.adj(L_k))] if adjoint else [])
        f = P
        for ci, w in zip(c, words):
            f = g(f + w * ci)
        return [f]
