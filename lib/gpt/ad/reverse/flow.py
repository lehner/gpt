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
# Flows (cotangents): what a node's gradient holds during a reverse pass.
#
#   None                  no flow yet (a zero that has not been built)
#   dense(value, owned)   a value: a plain field, number, array, tensor or
#                         series, or a node graph (a recorded pass); owned
#                         tells whether the gradient may update it in place
#                         (an adopted contribution is shared, e.g. both
#                         children of an add receive z's flow, and is copied
#                         before an in-place update)
#   scaled_identity(c, identity)
#                         exactly c times the (plain) identity of the node's
#                         type: a scalar flow broadcast back to a field by a
#                         reduction (trace, sum).  Built only when read, so
#                         consumers that exploit it (the stencil adjoints
#                         fold c into their weights) never allocate it
#   flow_list(elements)   the flow of a list node, per element a flow or None
#
# node.flow is the typed flow, node.gradient its value (built on reading).
# accumulate adds a contribution to a flow: None + r adopts r, scaled +
# scaled stays scaled, scaled + dense builds the field, lists add element by
# element.
#
import gpt as g
from gpt.ad.reverse.util import value_of, is_node, product, add, sub, recording


class dense:
    __slots__ = ("value", "owned")

    def __init__(self, value, owned):
        self.value = value
        self.owned = owned


class scaled_identity:
    __slots__ = ("c", "identity")

    def __init__(self, c, identity):
        self.c = c
        self.identity = identity


class flow_list:
    __slots__ = ("elements",)

    def __init__(self, elements):
        self.elements = elements


def scale(flow):
    # c if the flow is exactly c times the identity, else None
    return flow.c if isinstance(flow, scaled_identity) else None


def value(flow):
    # the value of a flow without building anything (None: no flow); a list
    # flow gives the list of its element values
    if flow is None:
        return None
    if isinstance(flow, flow_list):
        return [None if e is None else e.value for e in flow.elements]
    return flow.value


def wrap(v, container):
    # a value assigned as a gradient (the gradient owns it)
    if v is None:
        return None
    if container.tag[0] is list and isinstance(v, list):
        return flow_list([None if e is None else dense(e, True) for e in v])
    return dense(v, True)


def built(flow, container):
    # the flow with a scaled identity built into a field
    if isinstance(flow, scaled_identity):
        return accumulate(None, product(flow.identity, flow.c), 1, container)
    return flow


def owned(flow):
    # the flow with every adopted field copied (exclusively owned)
    if isinstance(flow, dense) and not flow.owned:
        return dense(g.copy(flow.value), True)
    if isinstance(flow, flow_list):
        return flow_list([None if e is None else owned(e) for e in flow.elements])
    return flow


def replaced(flow, v):
    # the flow with its value replaced by v (e.g. a conversion): a new field
    # is owned, the same object keeps its ownership
    if isinstance(flow, flow_list):
        return flow_list(
            [None if e is None else replaced(e, x) for e, x in zip(flow.elements, v)]
        )
    return dense(v, flow.owned or v is not flow.value)


def accumulate(cur, r, sign, container, adopt=True):
    # the flow cur + sign * r, for a contribution r (a value, a node graph, a
    # list of values, or a scaled_identity) to a gradient of the container:
    #   first contribution             adopted if it is a node graph of the
    #                                  same container while recording (graphs
    #                                  are immutable) or a plain field of the
    #                                  container (adopt=True), as not owned;
    #                                  otherwise assigned into a fresh
    #                                  (owned) field
    #   plain gradient +- plain term   in place if owned, else into a fresh
    #                                  field
    #   recording: a node on either    builds the (lazy) compute graph (a
    #   side                           plain side is a constant)
    #   not recording: a node term     the term graph is evaluated to a field
    #                                  (e.g. a node seed of a plain pass)
    #
    # Adopting is safe since a node never writes into its gradient after
    # passing it on: the backward pass runs in reverse topological order, so
    # all contributions to a node arrive before its own backward hands the
    # gradient to its children, after which it is released.  It also requires
    # that backward closures return fields they do not reuse.
    if isinstance(r, scaled_identity):
        if sign < 0:
            r = scaled_identity(-r.c, r.identity)
        if cur is None:
            return r
        if isinstance(cur, scaled_identity):
            return scaled_identity(cur.c + r.c, cur.identity)
        r, sign = product(r.identity, r.c), 1
    cur = built(cur, container)
    if container.tag[0] is list and isinstance(r, list):
        # a whole list flowing into a list node: element by element (None:
        # no flow into that element)
        assert cur is None or isinstance(cur, flow_list)
        elements = [None] * len(r) if cur is None else list(cur.elements)
        for i, x in enumerate(r):
            if x is not None:
                elements[i] = accumulate(elements[i], x, sign, container.tag[1], adopt)
        return flow_list(elements)
    assert not isinstance(cur, flow_list), "a list flow receives a list"
    graph = recording()
    if cur is None:
        if graph and is_node(r):
            if sign > 0 and r._container == container:
                return dense(r, True)
        elif container.tag[0] == g.lattice:
            r = value_of(r) if is_node(r) else r
            if (
                adopt
                and sign > 0
                and isinstance(r, g.lattice)
                and r.grid.obj == container.get_grid().obj
                and r.otype.__name__ == container.get_otype().__name__
            ):
                return dense(r, False)
            dst = g.lattice(container.get_grid(), container.get_otype())
            dst @= r if sign > 0 else -r
            return dense(dst, True)
        cur = dense(container.zero(), True)
    v = cur.value
    if graph and (is_node(v) or is_node(r)):
        return dense(add(v, r) if sign > 0 else sub(v, r), True)
    r = value_of(r) if is_node(r) else r
    if not cur.owned:
        return dense(g(v + r if sign > 0 else v - r), True)
    if sign > 0:
        v += r
    else:
        v -= r
    return dense(v, True)


def accum(n, r, sign=1, adopt=True):
    # accumulate sign * r into the flow of node n
    if n.flow is None and isinstance(n.value, g.ad.forward.series):
        n.zero_gradient()
    n.flow = accumulate(
        n.flow, r, sign, n._container, adopt
    )


def accum_element(n, i, r):
    # accumulate r into the flow of element i of list node n
    if n.flow is None:
        n.flow = flow_list([None] * len(n))
    n.flow.elements[i] = accumulate(
        n.flow.elements[i], r, 1, n._container.tag[1]
    )
