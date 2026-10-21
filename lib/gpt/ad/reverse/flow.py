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
#   pending_sum(partial, base, grid)
#                         a number's flow base + (the sum over the ranks of
#                         partial), partial the rank-local sum of its lattice
#                         contributions: a deferred reduction (plain passes
#                         only).  The reverse pass sets aside the nodes whose
#                         flow is pending (and the nodes below them), reduces
#                         all pending sums of the pass in one global sum per
#                         grid (flush), then processes the nodes set aside;
#                         so a pass needs one all-reduce instead of one per
#                         product of a number with a lattice
#
# A contribution (what a vjp returns) is a value, a list of values, a
# scaled_identity, or negative(contribution): its negative, subtracted without
# being built (e.g. the flow into y of x - y, also for a list).
#
# node.flow is the typed flow, node.gradient its value (built on reading).
# accumulate adds a contribution to a flow: None + r adopts r, scaled +
# scaled stays scaled, scaled + dense builds the field, lists add element by
# element.
#
import gpt as g
import numpy as np
from gpt.ad.reverse.util import value_of, is_node, recording


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


class pending_sum:
    __slots__ = ("partial", "base", "grid", "result")

    def __init__(self, partial, base, grid):
        self.partial = partial
        self.base = base
        self.grid = grid
        # (the value, once reduced)
        self.result = None


# the nodes with a pending flow, one list per running reverse pass (a pass
# nested in a backward reduces its own); defer: switch (for comparisons)
_pending = []
defer = True


def begin_pass():
    _pending.append([])


def end_pass():
    # (the reverse pass reduces its pending sums before it ends; after an
    # exception nothing is reduced: a global sum is collective)
    _pending.pop()


def deferring():
    # whether a plain reverse pass runs (sums into numbers may be deferred)
    return defer and bool(_pending) and not recording()


def flush():
    # reduce the pending flows of the running pass: one global sum per grid
    nodes = _pending[-1] if _pending else []
    if not nodes:
        return
    by_grid = {}
    for n in nodes:
        f = n.flow
        if isinstance(f, pending_sum) and f.result is None:
            by_grid.setdefault(f.grid.obj, (f.grid, []))[1].append(n)
    for grid, group in by_grid.values():
        total = grid.globalsum(np.array([n.flow.partial for n in group], dtype=np.complex128))
        for n, t in zip(group, total):
            f = n.flow
            f.result = f.base + complex(t)
            n.flow = dense(f.result, True)
    nodes.clear()


class negative:
    # the contribution -value (see above)
    __slots__ = ("value",)

    def __init__(self, value):
        self.value = value


def scale(flow):
    # c if the flow is exactly c times the identity, else None
    return flow.c if isinstance(flow, scaled_identity) else None


def value(flow):
    # the value of a flow without building anything (None: no flow); a list
    # flow gives the list of its element values
    if flow is None:
        return None
    if isinstance(flow, pending_sum):
        # (read before the pass reduced it: reduce now)
        if flow.result is None:
            flush()
        return flow.result
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
        return accumulate(None, flow.identity * flow.c, 1, container)
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
    #                                  are immutable; a negative lattice one
    #                                  as -1 * graph) or a plain field of the
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
    if isinstance(r, negative):
        r, sign = r.value, -sign
    if isinstance(r, pending_sum) or isinstance(cur, pending_sum):
        return _accumulate_pending(cur, r, sign)
    if isinstance(r, scaled_identity):
        if sign < 0:
            r = scaled_identity(-r.c, r.identity)
        if cur is None:
            return r
        if isinstance(cur, scaled_identity):
            return scaled_identity(cur.c + r.c, cur.identity)
        r, sign = r.identity * r.c, 1
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
            if r._container == container:
                if sign > 0:
                    return dense(r, True)
                if container.tag[0] == g.lattice:
                    # (-1 * r rather than zero - r: no zero field held by
                    # the recorded graph)
                    r_neg = (-1.0) * r
                    if r_neg._container == container:
                        return dense(r_neg, True)
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
        return dense(v + r if sign > 0 else v - r, True)
    r = value_of(r) if is_node(r) else r
    if not cur.owned:
        return dense(g(v + r if sign > 0 else v - r), True)
    if sign > 0:
        v += r
    else:
        v -= r
    return dense(v, True)


def _accumulate_pending(cur, r, sign):
    # a number flow with a pending part (see pending_sum)
    if isinstance(cur, dense):
        cur = pending_sum(0.0, cur.value, r.grid)
    if isinstance(r, pending_sum):
        if cur is None:
            return pending_sum(sign * r.partial, r.base * sign, r.grid)
        assert cur.grid.obj == r.grid.obj, "pending sums of one number on one grid"
        return pending_sum(cur.partial + sign * r.partial, cur.base + sign * r.base, cur.grid)
    r = value_of(r) if is_node(r) else r
    return pending_sum(cur.partial, cur.base + sign * r, cur.grid)


def accum(n, r, sign=1, adopt=True):
    # accumulate sign * r into the flow of node n
    if n.flow is None and isinstance(n.value, g.ad.forward.series):
        n.zero_gradient()
    was_pending = isinstance(n.flow, pending_sum)
    n.flow = accumulate(n.flow, r, sign, n._container, adopt)
    if not was_pending and isinstance(n.flow, pending_sum):
        _pending[-1].append(n)
