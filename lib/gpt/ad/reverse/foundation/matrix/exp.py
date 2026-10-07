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
# Plain evaluation: gpt.core.foundation.lattice.matrix.exp.derivative (exp
# of a multi-dual number, one compiled kernel per D_k).
#
import gpt as g
from gpt.ad.reverse.util import is_node, nodify, value_of


def _plain_base(x):
    # x = L or x = adj(L) for a plain lattice L: (L, adjoint?); else (None, None).
    # Flows at X^dag are the lazy expression adj(X) (also when built at node
    # level, where adj stays unevaluated), and adj(adj(X)) is X
    if isinstance(x, g.lattice):
        return x, False
    if isinstance(x, g.expr) and x.unary == g.expr_unary.NONE and len(x.val) == 1:
        coef, term = x.val[0]
        if coef == 1.0 and len(term) == 1 and isinstance(term[0][1], g.lattice):
            unary = term[0][0]
            if unary == g.factor_unary.NONE:
                return term[0][1], False
            if unary == g.factor_unary.ADJ:
                return term[0][1], True
    return None, None


class _tower:
    # the plain X shared by all D_k nodes of one exp tower (the node exp(X)
    # and the flows built from it, at any nesting depth): every plain D_k is
    # evaluated at X or X^dag, which have the same scaling s, so s and the
    # materialized X^dag are computed once.  X is identified by identity; the
    # user-created root resets the tower whenever it recomputes its value, so
    # a leaf modified in place between evaluations is never served stale
    def __init__(self):
        self.reset()

    def reset(self):
        self.x = None
        self.s = None
        self.xdag = None

    def plain(self, x, h):
        base, dag = _plain_base(x)
        if base is None:
            return g.lattice.foundation.matrix.exp.derivative(x, h)
        if base is not self.x:
            self.x = base
            self.s = g.lattice.foundation.matrix.exp.scaling(base)
            self.xdag = None
        if dag:
            if self.xdag is None:
                self.xdag = g(g.adj(base))
            base = self.xdag
        return g.lattice.foundation.matrix.exp.derivative(base, h, self.s)


def derivative(x, h, tower=None):
    # D_k(x; h_1..h_k); plain values run the fused kernel, node values build a
    # node whose backward is again a D (one level down), in the same tower
    args = [x] + list(h)
    if not any(is_node(a) for a in args):
        # (fused kernels live in the lattice foundation)
        return (tower or _tower()).plain(x, h)

    args = nodify(*args) if len(args) > 1 else (args[0],)
    xn, hn = args[0], list(args[1:])
    k = len(hn)
    root = tower is None
    if root:
        tower = _tower()

    def _forward():
        if root:
            tower.reset()
        return derivative(value_of(xn), [value_of(c) for c in hn], tower)

    def _backward_x(z):
        return (
            1,
            derivative(g.adj(value_of(xn)), [g.adj(value_of(c)) for c in hn] + [z.gradient], tower),
        )

    def _backward_h(i):
        def _b(z):
            return (
                1,
                derivative(
                    g.adj(value_of(xn)),
                    [z.gradient if j == i else g.adj(value_of(c)) for j, c in enumerate(hn)],
                    tower,
                ),
            )

        return _b

    return g.ad.reverse.node_op(
        tuple(args),
        _forward,
        (_backward_x,) + tuple(_backward_h(i) for i in range(k)),
        xn._container,
        f"exp_d{k}",
    )


def function(x):
    if x._container.tag[0] != g.lattice:
        # non-lattice (tensor / scalar) values: the generic scaling-and-
        # squaring Taylor series of the forward AD, as a graph of node ops
        return g.ad.forward.foundation.matrix.exp.function(x)
    return derivative(x, [])
