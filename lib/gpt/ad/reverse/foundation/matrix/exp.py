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


def _plain(x, h):
    # the fused multi-dual kernels live in the lattice foundation
    return g.lattice.foundation.matrix.exp.derivative(x, h)


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
