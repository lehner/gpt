#
#    GPT - Grid Python Toolkit
#    Copyright (C) 2023-2026  Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)
#                  2026       Christopher Kelly
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
from gpt.ad.reverse.util import (
    container,
    get_unary_container,
    get_container,
    product,
    accum,
    value_of,
    is_node,
    nodify,
)
import gpt.ad.reverse.foundation.matrix


def inner_product(x, y, n_block, use_accelerator):
    assert len(x) == 1 and len(y) == 1 and n_block == 1
    # the contraction is symmetric in its arguments: plain operands are
    # promoted to constant nodes (a node's children must be nodes)
    x, y = nodify(x[0], y[0])

    def _forward():
        vx, vy = nodify(value_of(x), value_of(y))
        return g.inner_product(vx, vy, n_block, use_accelerator)

    # z = adj(x) y   ->   x = y adj(z)   and   y = x z
    return {
        (0, 0): g.ad.reverse.node_op(
            (x, y),
            _forward,
            (
                lambda z: (1, product(value_of(y), g.adj(z.gradient))),
                lambda z: (1, product(value_of(x), z.gradient)),
            ),
            container(complex),
            "inner_product",
        )
    }


def norm2(x):
    assert len(x) == 1
    return [g.inner_product(x, x)[0, 0]]


def cshift(x, direction, displacement, none):
    assert none is None

    return g.ad.reverse.node_op(
        (x,),
        lambda: g.cshift(value_of(x), direction, displacement),
        (lambda z: (1, g.cshift(z.gradient, direction, -displacement)),),
        x._container,
        "cshift(" + str(direction) + ", " + str(displacement) + ")",
    )


def adj(x):
    return g.ad.reverse.node_op(
        (x,),
        lambda: g.adj(value_of(x)),
        (lambda z: (1, g.adj(z.gradient)),),
        x._container,
        "adj",
    )


def trace(x, t):
    z_container = get_unary_container(x._container, lambda v: g.trace(v, t))

    return g.ad.reverse.node_op(
        (x,),
        lambda: g.trace(value_of(x), t),
        (lambda z: (1, product(g.identity(value_of(x)), z.gradient)),),
        z_container,
    )


def sum(x):
    return g.ad.reverse.node_op(
        (x,),
        lambda: g.sum(value_of(x)),
        (lambda z: (1, product(g.identity(value_of(x)), z.gradient)),),
        x._container.lattice_to_tensor(),
    )


def component_simple_map(operator, numpy_operator, extra_params, first, second):
    if operator == "relu":
        assert second is None
        return g.ad.reverse.transform.relu(first, a=extra_params["a"])
    elif operator == "sin":
        assert second is None
        return g.ad.reverse.transform.sin(first)
    elif operator == "cos":
        assert second is None
        return g.ad.reverse.transform.cos(first)
    raise Exception(f"component-wise operator {operator} not implemented in rev-AD")


def component_multiply(a, b):
    """Element-wise product; node-aware, with the plain case kept on the
    lattice foundation's component kernel (which covers more otypes than
    plain multiplication)"""
    if not is_node(a) and not is_node(b):
        return g.lattice.foundation.component_multiply(a, b)
    return product(a, b)


def _group_conversion(src, dsrc, method):
    # dispatch on the perturbation's otype conversion method (infinitesimal_to_
    # cartesian or cartesian_to_infinitesimal), as in the lattice and forward-AD
    # foundations; node.value may be None for unevaluated (or freed) nodes
    if gpt.util.is_num(dsrc.value) or isinstance(dsrc.value, np.ndarray):
        return dsrc
    if is_node(dsrc):
        # a nested gradient is a lazy compute graph; the otype conversion is
        # linear in the gradient and runs as graph operations (including the
        # container otype update); containers without an otype, or otypes
        # without the conversion, pass through
        try:
            otype = dsrc.otype
        except Exception:
            return dsrc
        if not hasattr(otype, method):
            return dsrc
        return getattr(otype, method)(src, dsrc)
    return getattr(dsrc.otype, method)(src, dsrc)


def infinitesimal_to_cartesian(src, dsrc):
    return _group_conversion(src, dsrc, "infinitesimal_to_cartesian")


def cartesian_to_infinitesimal(src, dsrc):
    return _group_conversion(src, dsrc, "cartesian_to_infinitesimal")


def identity(x):
    def _forward():
        # a plain (expr) value at the bottom of a (lazy) chain has no
        # foundation to dispatch on; evaluate it to a field first
        v = value_of(x)
        if isinstance(v, g.expr):
            v = g(v)
        return g.identity(v)

    return g.ad.reverse.node_op(
        (x,),
        _forward,
        (None,),
        x._container,
        "identity(" + str(x._container) + ")",
    )


def astype(x, y):
    z_container = x._container.copy()
    z_container.set_otype(y)

    return g.ad.reverse.node_op(
        (x,),
        lambda: g.astype(g(value_of(x)), y),
        (lambda z: (1, z.gradient),),
        z_container,
        "astype(" + str(x._container) + "," + str(y) + ")",
    )


def cshift_plan_add(self, fields, displacements):
    indices = {}
    for d in displacements:
        indices[d] = self.index
        self.index += 1
    self.indices.append(indices)
    return indices


def cshift_plan_execute(self):
    def _executer(first, second=None):
        assert second is None
        ret = []
        for i, displacements in enumerate(self.displacements):
            for d in displacements:
                ret.append(first[i])
                for dir, disp in enumerate(d):
                    if disp != 0:
                        ret[-1] = g.cshift(ret[-1], dir, disp)
        return ret

    return _executer


def group_inner_product(left, right):
    # inner product over group's real vector space; symmetric in its
    # arguments, so plain operands are promoted to constant nodes (a node's
    # children must be nodes)
    left, right = nodify(left, right)
    left_type = left.otype
    return left_type.inner_product(left, right)


def where(first, second, third, fourth):
    assert fourth is None
    question = first
    yes = second
    no = third

    z_container = yes._container

    # node-aware: nodify the operands so a nested pair (a nested value or a
    # node flow) routes to the rev-AD where instead of the plain foundation,
    # which cannot build a lattice from a node.

    def _forward():
        vy, vn = nodify(value_of(yes), value_of(no))
        return g.where(question, vy, vn)

    return g.ad.reverse.node_op(
        (yes, no),
        _forward,
        (
            lambda z: (1, g.where(question, *nodify(z.gradient, yes._container.zero()))),
            lambda z: (1, g.where(question, *nodify(no._container.zero(), z.gradient))),
        ),
        z_container,
        "where(" + str(yes._container) + ")",
    )
