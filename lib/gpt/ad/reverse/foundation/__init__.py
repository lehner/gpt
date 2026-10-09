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
    get_container,
    get_unary_container,
    value_of,
    is_node,
    nodify,
)
from gpt.ad.reverse import flow as flows
from gpt.ad.reverse.primitive import primitive
from gpt.ad.reverse import tangent
import gpt.ad.reverse.foundation.matrix
import gpt.ad.reverse.foundation.stencil
import gpt.ad.reverse.foundation.local_stencil


def _plain_inner_product(x, y, n_block, use_accelerator):
    if gpt.util.is_num(x) and gpt.util.is_num(y):
        # support for "0d vectors"
        return gpt.adj(x) * y
    return g.inner_product(x, y, n_block, use_accelerator)


# z = adj(x) y   ->   x = y adj(z)   and   y = x z
_inner_product = primitive(
    "inner_product",
    _plain_inner_product,
    lambda x, y, **static: container(complex),
    vjp=lambda i, flow, x, y, **static: (
        y * g.adj(flow) if i == 0 else x * flow
    ),
    reads=((1,), (0,)),
)


def inner_product(x, y, n_block, use_accelerator):
    assert len(x) == 1 and len(y) == 1 and n_block == 1
    # the contraction is symmetric in its arguments: plain operands are
    # promoted to constant nodes (a node's children must be nodes)
    return {
        (0, 0): _inner_product.node(
            x[0], y[0], n_block=n_block, use_accelerator=use_accelerator
        )
    }


def norm2(x):
    assert len(x) == 1
    return [g.inner_product(x, x)[0, 0]]


def _same(x, **static):
    # the container of a map that keeps the type
    return x


_cshift = primitive(
    "cshift",
    lambda x, direction, displacement: g.cshift(x, direction, displacement),
    _same,
    vjp=lambda i, flow, x, direction, displacement: g.cshift(flow, direction, -displacement),
    reads=((),),
)


def cshift(x, direction, displacement, none):
    assert none is None
    return _cshift.node(x, direction=direction, displacement=displacement)


_adj = primitive(
    "adj",
    lambda x: g.adj(x),
    _same,
    vjp=lambda i, flow, x: g.adj(flow),
    reads=((),),
    jvp=tangent.linear(g.adj),
    involution=True,
)


def adj(x):
    return _adj.node(x)


def _reduction_identity(x):
    # identity(x) for the backward of a reduction.  Only the type of x is
    # needed, never its value: for a lattice on a full grid it is taken from
    # the container (a plain constant; evaluating x would cost its forward,
    # and in a recorded pass add x to the flow's graph)
    # (a forward-AD series value keeps its own identity type)
    c = x._container
    if (
        c.tag[0] is g.lattice
        and c.get_grid().cb.n == 1
        and not isinstance(x.value, g.ad.forward.series)
    ):
        return g.identity_constant(g.lattice(c.get_grid(), c.get_otype()))
    return g.identity_constant(value_of(x))


def _reduction_vjp(z, needed, x, **static):
    # a sum-like reduction (trace/sum).  Its adjoint broadcasts the flow back
    # to x's lattice via identity(x) (conjugate-linear in the flow).  If that
    # flow is a scalar c, or itself c times the identity, the flow into a
    # plain lattice x is exactly c times the identity: a scaled_identity flow
    # (see flow.py), built into a field only if a consumer reads it.  (The
    # type of x, not its value: the node x itself)
    (x,) = z._children
    c = flows.scale(z.flow)
    if c is None and g.util.is_num(z.flow.value):
        c = complex(z.flow.value)
    if c is not None and x._container.tag[0] is g.lattice:
        return {0: flows.scaled_identity(c, _reduction_identity(x))}
    return {0: _reduction_identity(x) * z.gradient}


_trace = primitive(
    "trace",
    lambda x, t: g.trace(x, t),
    lambda x, t: get_unary_container(x, lambda v: g.trace(v, t), ("trace", t)),
    joint_vjp=_reduction_vjp,
    reads=((),),
    # (the trace is site-local and linear; the sum over sites is not)
    jvp=tangent.linear(lambda v, t: trace(v, t)),
)


def trace(x, t):
    return _trace.node(x, t=t)


_sum = primitive(
    "sum",
    lambda x: g.sum(x),
    lambda x: x.lattice_to_tensor(),
    joint_vjp=_reduction_vjp,
    reads=((),),
)


def sum(x):
    return _sum.node(x)


# the componentwise maps that are node primitives (in transform.py, by name)
_component_maps = {"relu", "drelu", "sin", "cos", "real", "imag"}


def component_simple_map(operator, numpy_operator, extra_params, first, second):
    assert second is None
    if operator not in _component_maps:
        raise Exception(f"component-wise operator {operator} not implemented in rev-AD")
    return getattr(g.ad.reverse.transform, operator)(first, **extra_params)


def component_multiply(a, b):
    # element-wise product, node-aware (plain lattices on the lattice
    # foundation's component kernel, which covers more otypes than plain
    # multiplication)
    return g.ad.reverse.transform.multiply(a, b)


def _project(v, name):
    return getattr(g.qcd.gauge.project, name)(v)


# a real-linear projection P that is self-adjoint w.r.t. Re tr(a^dag b) (the
# traceless (anti-)hermitian parts): the flow into x is P(flow), so the
# backward is again the projection (a projection node in a recorded pass),
# and the node replaces the graph of its adj, sums, trace and identity
_projection = primitive(
    "projection",
    _project,
    _same,
    vjp=lambda i, flow, x, name: _project(flow, name),
    reads=((),),
    jvp=tangent.linear(_project),
)


def traceless_anti_hermitian(x):
    return _projection.node(x, name="traceless_anti_hermitian")


def traceless_hermitian(x):
    return _projection.node(x, name="traceless_hermitian")


def _group_conversion(src, dsrc, method):
    # dispatch on the perturbation's otype conversion method (infinitesimal_to_
    # cartesian or cartesian_to_infinitesimal), as in the lattice and forward-AD
    # foundations; node.value may be None for unevaluated (or freed) nodes
    if gpt.util.is_num(dsrc.value) or isinstance(dsrc.value, np.ndarray):
        return dsrc
    if is_node(dsrc):
        # a recorded gradient is a lazy compute graph; the otype conversion is
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


def _plain_identity(x):
    # a plain (expr) value at the bottom of a (lazy) chain has no foundation
    # to dispatch on; evaluate it to a field first
    if isinstance(x, g.expr):
        x = g(x)
    # (node values are never modified, so the identity can be shared)
    return g.identity_constant(x)


# the identity of x's type: a constant (no flow, no tangent)
_identity = primitive(
    "identity",
    _plain_identity,
    _same,
    vjp=lambda i, flow, x: None,
    reads=((),),
    jvp=tangent.constant,
)


def identity(x):
    return _identity.node(x)


def _astype_container(x, otype):
    c = x.copy()
    c.set_otype(otype)
    return c


_astype = primitive(
    "astype",
    lambda x, otype: g.astype(g(x), otype),
    _astype_container,
    vjp=lambda i, flow, x, otype: flow,
    reads=((),),
    jvp=tangent.linear(lambda v, otype: astype(v, otype)),
)


def astype(x, y):
    return _astype.node(x, otype=y)


def group_inner_product(left, right):
    # inner product over group's real vector space; symmetric in its
    # arguments, so plain operands are promoted to constant nodes (a node's
    # children must be nodes)
    left, right = nodify(left, right)
    left_type = left.otype
    return left_type.inner_product(left, right)


def _where_vjp(i, flow, yes, no, question, branch):
    # the flow where the branch is taken, zero elsewhere (branch: the
    # container of both branches).  Node-aware: a recorded flow routes to the
    # rev-AD where, not to the plain foundation, which cannot build a lattice
    # from a node
    if i == 0:
        return g.where(question, *nodify(flow, branch.zero()))
    return g.where(question, *nodify(branch.zero(), flow))


_where = primitive(
    "where",
    lambda yes, no, question, branch: g.where(question, yes, no),
    lambda yes, no, question, branch: yes,
    vjp=_where_vjp,
    reads=((), ()),
)


def where(first, second, third, fourth):
    assert fourth is None
    return _where.node(second, third, question=first, branch=get_container(second))
