#
#    GPT - Grid Python Toolkit
#    Copyright (C) 2023  Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)
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
import operator


def otype_compatible(a, b):
    # otype-level check (distinct from container.accumulate_compatible, which
    # also compares the lattice grid); resolves data aliases before comparing
    if a == complex:
        a = g.ot_singlet()
    if b == complex:
        b = g.ot_singlet()
    if a.data_alias is not None:
        a = a.data_alias()
    if b.data_alias is not None:
        b = b.data_alias()
    return a.__name__ == b.__name__


class container:
    def __init__(self, *tag):
        if isinstance(tag[0], tuple):
            tag = tag[0]
            if tag[1] is None:
                tag = [complex]
            elif tag[0] is None:
                tag = [g.tensor, tag[1]]
            else:
                tag = [g.lattice, tag[0], tag[1]]

        self.tag = tag

        if self.tag[0] == g.tensor:
            otype = self.tag[1]
            while otype.data_alias is not None:
                otype = otype.data_alias()
            if otype.__name__ == "ot_singlet":
                self.tag = [complex]

    def copy(self):
        return container(*[x for x in self.tag])
        
    def is_field(self):
        if self.tag[0] is list:
            return self.tag[1].is_field()
        return self.tag[0] == g.lattice

    def representative(self):
        if self.tag[0] is list:
            _, elem, n = self.tag
            return [elem.representative() for _ in range(n)]
        return self.tag[0](*self.tag[1:])

    def lattice_to_tensor(self):
        if self.tag[0] is list:
            return container(list, self.tag[1].lattice_to_tensor(), self.tag[2])
        assert self.tag[0] == g.lattice
        return container(g.tensor, self.tag[2])

    def get_grid(self):
        if self.tag[0] is list:
            return self.tag[1].get_grid()
        if len(self.tag) > 2:
            return self.tag[1]
        raise Exception("Container does not have a grid")

    def get_otype(self):
        if self.tag[0] is list:
            return self.tag[1].get_otype()
        if len(self.tag) > 1:
            return self.tag[-1]
        raise Exception("Container does not have an otype")

    def set_otype(self, otype):
        if self.tag[0] is list:
            self.tag[1].set_otype(otype)
            return
        if len(self.tag) > 1:
            self.tag = list(self.tag[:-1]) + [otype]
        else:
            raise Exception("Container does not have an otype")

    def accumulate_compatible(self, other):
        if self.tag[0] is list or other.tag[0] is list:
            if self.tag[0] is list and other.tag[0] is list:
                return self.tag[2] == other.tag[2] and self.tag[1].accumulate_compatible(
                    other.tag[1]
                )
            return False
        if len(self.tag) > 1 and len(other.tag) > 1:
            if len(self.tag) != len(other.tag):
                return False
            if len(self.tag) > 2:
                if self.get_grid().obj != other.get_grid().obj:
                    return False
            a = self.get_otype()
            b = other.get_otype()
            if a.data_alias is not None:
                a = a.data_alias()
            if b.data_alias is not None:
                b = b.data_alias()
            return a.__name__ == b.__name__

        return self.__eq__(other)

    def zero(self):
        r = self.representative()
        if isinstance(r, list):
            return [self._zero_one(e) for e in r]
        return self._zero_one(r)

    def _zero_one(self, r):
        if isinstance(r, g.lattice):
            r[:] = 0
        elif isinstance(r, (g.tensor, np.ndarray)):
            r *= 0
        elif isinstance(r, complex):
            r = 0.0
        else:
            raise Exception("Unknown type")
        return r

    def __eq__(self, other):
        return str(self) == str(other)

    def __str__(self):
        if self.tag[0] is list:
            return "list[%d](%s)" % (self.tag[2], str(self.tag[1]))
        r = str(self.tag[0].__name__)
        if len(self.tag) > 1:
            r = r + ";" + self.tag[-1].__name__
        if len(self.tag) == 3:
            r = r + ";" + str(self.tag[1].obj)
        return r


def get_container(x):
    if isinstance(x, g.expr):
        x = g(x)
    if isinstance(x, g.ad.reverse.node_base):
        return get_container(x.value)
    elif isinstance(x, g.ad.forward.series):
        for t in x.terms:
            return get_container(x[t])
        raise Exception("Empty series")
    elif isinstance(x, list):
        # a uniform list of fields (e.g. the 4 gauge links) is one list node
        if len(x) == 0:
            raise Exception("empty list has no container")
        elem = get_container(x[0])
        for e in x[1:]:
            ce = get_container(e)
            if not ce.accumulate_compatible(elem):
                raise Exception(
                    "list elements must be mutually compatible: %s vs %s" % (elem, ce)
                )
        return container(list, elem, len(x))
    elif isinstance(x, g.lattice):
        return container(g.lattice, x.grid, x.otype)
    elif isinstance(x, g.tensor):
        return container(g.tensor, x.otype)
    elif isinstance(x, np.ndarray):
        return container(np.ndarray, x.shape, x.dtype)
    elif g.util.is_num(x):
        return container(complex)
    else:
        raise Exception(f"Unknown object type {type(x)}")


def is_node(x):
    # deferred reference: node_base lives in node.py, which imports util
    return isinstance(x, g.ad.reverse.node_base)


def nodify(*args):
    # promote plain operands to constant nodes so they can be combined with
    # node-typed (lazy) values without leaving the node world.  All-plain
    # input passes through unchanged, so plain arithmetic keeps its exact
    # (non-node) dispatch.  Returns the (possibly wrapped) argument for a
    # single argument, otherwise a tuple.
    if any(is_node(a) for a in args):
        args = tuple(
            a if is_node(a) else g.ad.reverse.node_base(a, with_gradient=False) for a in args
        )
    return args[0] if len(args) == 1 else args


def value_of(x):
    # the raw (possibly node-typed) value of x; a node that was freed after a
    # previous pass (value = None) is re-evaluated in place.  The value is NOT
    # resolved across (nested) nodes: node-typed values carry the
    # differentiation dependencies needed by deeper reverse passes
    if x.value is None and x._forward is not None:
        x.value = x._forward()
    return x.value


def value_depth_static(x):
    # number of nested node levels below x, WITHOUT forcing a forward
    # evaluation.  This is the only depth measure: resolving the depth by
    # EVALUATING computed nodes (the obvious `while is_node(x): x = value_of(x)`)
    # is a trap that has bitten twice -- before a graph is first run it caches
    # a value that node.forward (which only recomputes values that are None)
    # then reuses instead of rebuilding it from the updated leaves, and inside
    # backward() it re-materializes fields that pass has just freed.  Measuring
    # a depth is never a reason to run a forward closure.
    #
    # This runs on every node of every backward pass (zero_gradient), so it
    # walks a SINGLE path: down the .value chain while a value is available,
    # and sideways through one child of a computed node whose value is not
    # built yet.  One child settles it because a node op combines operands of
    # equal depth; a gradient-carrying one is authoritative, since a constant
    # promoted by nodify may be shallower.  Taking the max over ALL children
    # instead re-traverses shared subgraphs and is exponential in a DAG --
    # that cost 82M calls and a 24x slowdown of the flows.py force.
    depth = 0
    while is_node(x):
        v = x.value
        if v is not None:
            depth += 1
            x = v
            continue
        # computed node, value not built yet: same depth as its children
        nxt = None
        for c in x._children:
            if c.with_gradient:
                nxt = c
                break
        if nxt is not None:
            x = nxt
            continue
        if len(x._children) == 0:
            return depth + 1
        # a pure-constant expression: no child is authoritative, take the
        # deepest (constants may legitimately differ in depth).  Rare, and
        # never on the hot path.
        return depth + max(value_depth_static(c) for c in x._children)
    return depth


def _binop(a, b, op):
    # node-aware binary operation: if either side is a node, both stay in the
    # node world (plain operands are promoted to constant nodes); otherwise
    # it dispatches exactly as plain arithmetic
    a, b = nodify(a, b)
    return op(a, b)


def product(a, b):
    return _binop(a, b, operator.mul)


def add(a, b):
    return _binop(a, b, operator.add)


def sub(a, b):
    return _binop(a, b, operator.sub)


def div(a, b):
    return _binop(a, b, operator.truediv)


def adopt_zero(current, term):
    # May `term` simply REPLACE `current` as a gradient?  True when current is
    # the structural zero that zero_gradient just installed and term is a node
    # graph of the very same container.  Adding to that zero would build an
    # add node whose zero operand is evaluated (a field add) and then
    # differentiated (a backward visit into a dead leaf, plus its own
    # zero_gradient and infinitesimal_to_cartesian) -- all to add nothing.
    # The container must match exactly: otype is load-bearing at the leaf
    # conversion, and add() would have kept current's container.
    return (
        is_node(current)
        and current._pristine_zero
        and is_node(term)
        and current._container == term._container
    )


def accum(n, r, sign=1):
    # accumulate sign * r into n.gradient:
    #   plain gradient +- plain term   in place
    #   plain gradient +- node term    the term graph is linear in the flow,
    #                                  so it is evaluated to a field, keeping
    #                                  the result in the plain world as in
    #                                  single-pass AD
    #   node gradient  +- plain/node   builds the (lazy) compute graph; a
    #   term                     subtraction with incompatible containers is
    #                                  evaluated to a field
    if is_node(n.gradient):
        if sign > 0 and adopt_zero(n.gradient, r):
            n.gradient = r
        elif sign > 0:
            n.gradient = add(n.gradient, r)
        elif is_node(r) and n.gradient._container != r._container:
            n.gradient = value_of(n.gradient) - value_of(r)
        else:
            n.gradient = sub(n.gradient, r)
    elif is_node(r):
        if sign > 0:
            n.gradient += value_of(r)
        else:
            n.gradient -= value_of(r)
    elif sign > 0:
        n.gradient += r
    else:
        n.gradient -= r


def get_mul_container(x, y):
    rx = x.representative()
    ry = y.representative()
    return container((g.expr(rx) * g.expr(ry)).container())


def get_div_container(x, y):
    assert isinstance(y.representative(), complex)
    return x


def get_unary_container(x, unary):
    rx = x.representative()
    return container(g.expr(unary(rx)).container())


def convert_container(v, x, y, operand):
    rx = x.representative()
    ry = y.representative()

    r = g.expr(operand(rx, ry))
    c = container(r.container())

    if v._container.accumulate_compatible(c):
        return v

    # conversions from tensor to matrix
    backward_sum = False
    backward_spin_trace = False
    backward_color_trace = False
    backward_trace = False

    if v._container.tag[0] != g.lattice and c.tag[0] == g.lattice:
        backward_sum = True

    # now check otypes
    if v._container.tag[-1].__name__ != c.tag[-1].__name__:
        rhs_otype = c.tag[-1]
        lhs_otype = v._container.tag[-1]

        if rhs_otype.spintrace[2] is not None:
            rhs_spintrace_otype = rhs_otype.spintrace[2]()
            if otype_compatible(lhs_otype, rhs_spintrace_otype):
                backward_spin_trace = True
                rhs_otype = rhs_spintrace_otype
            elif rhs_spintrace_otype.colortrace[2] is not None:
                rhs_trace_otype = rhs_spintrace_otype.colortrace[2]()
                if otype_compatible(lhs_otype, rhs_trace_otype):
                    backward_trace = True
                    rhs_otype = rhs_trace_otype
        if rhs_otype.colortrace[2] is not None:
            rhs_colortrace_otype = rhs_otype.colortrace[2]()
            if otype_compatible(lhs_otype, rhs_colortrace_otype):
                backward_color_trace = True
                rhs_otype = rhs_colortrace_otype

        if not otype_compatible(rhs_otype, lhs_otype):
            raise Exception(
                "Conversion incomplete:" + rhs_otype.__name__ + ":" + lhs_otype.__name__
            )

    assert backward_trace or backward_color_trace or backward_spin_trace or backward_sum

    def _forward():
        # v.value may be a (lazy) node or None if v was freed after a
        # previous pass
        return value_of(v)

    def _backward(z):
        if v.with_gradient:
            gradient = z.gradient

            if backward_trace:
                gradient = g.trace(gradient)

            if backward_color_trace:
                gradient = g.color_trace(gradient)

            if backward_spin_trace:
                gradient = g.spin_trace(gradient)

            if backward_sum:
                gradient = g.sum(gradient)

            if (backward_trace or backward_color_trace or backward_spin_trace) and not is_node(
                gradient
            ):
                gradient = g(gradient)

            accum(v, gradient)

    return g.ad.reverse.node_base(
        _forward,
        _backward,
        (v,),
        _container=c,
        _tag="change to " + str(c) + " from " + str(v._container),
    )
