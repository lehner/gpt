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


def accumulate_compatible(a, b):
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
        return self.tag[0] == g.lattice

    def representative(self):
        return self.tag[0](*self.tag[1:])

    def lattice_to_tensor(self):
        assert self.tag[0] == g.lattice
        return container(g.tensor, self.tag[2])

    def get_grid(self):
        if len(self.tag) > 2:
            return self.tag[1]
        raise Exception("Container does not have a grid")

    def get_otype(self):
        if len(self.tag) > 1:
            return self.tag[-1]
        raise Exception("Container does not have an otype")

    def set_otype(self, otype):
        if len(self.tag) > 1:
            self.tag = list(self.tag[:-1]) + [otype]
        else:
            raise Exception("Container does not have an otype")

    def accumulate_compatible(self, other):
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


def value_of(x):
    # the raw (possibly node-typed) value of x; a node that was freed after a
    # previous pass (value = None) is re-evaluated in place.  The value is NOT
    # resolved across (nested) nodes: node-typed values carry the
    # differentiation dependencies needed by deeper reverse passes
    if x.value is None and x._forward is not None:
        x.value = x._forward()
    return x.value


def _promote(a, b):
    # wrap plain operands as constant nodes so node-typed (lazy) values can be
    # combined with them without leaving the node world
    if not isinstance(a, g.ad.reverse.node_base):
        a = g.ad.reverse.node_base(a, with_gradient=False)
    if not isinstance(b, g.ad.reverse.node_base):
        b = g.ad.reverse.node_base(b, with_gradient=False)
    return a, b


def product(a, b):
    # node-aware product, preserving operand order
    if isinstance(a, g.ad.reverse.node_base) or isinstance(b, g.ad.reverse.node_base):
        a, b = _promote(a, b)
        return a * b
    return a * b


def add(a, b):
    if isinstance(a, g.ad.reverse.node_base) or isinstance(b, g.ad.reverse.node_base):
        a, b = _promote(a, b)
        return a + b
    return a + b


def sub(a, b):
    if isinstance(a, g.ad.reverse.node_base) or isinstance(b, g.ad.reverse.node_base):
        a, b = _promote(a, b)
        return a - b
    return a - b


def div(a, b):
    if isinstance(a, g.ad.reverse.node_base) or isinstance(b, g.ad.reverse.node_base):
        a, b = _promote(a, b)
        return a / b
    return a / b


def accum(n, r):
    # accumulate r into n.gradient; in place for plain values, node-aware
    # otherwise.  A node term into a plain gradient is evaluated to a field
    # (the term graph is linear in the flow), which keeps the result in the
    # plain world as in single-pass AD
    if isinstance(n.gradient, g.ad.reverse.node_base):
        n.gradient = add(n.gradient, r)
    elif isinstance(r, g.ad.reverse.node_base):
        n.gradient += value_of(r)
    else:
        n.gradient += r


def accum_sub(n, r):
    # accumulate -r into n.gradient (see accum); node gradients with
    # incompatible containers are evaluated to fields
    if isinstance(n.gradient, g.ad.reverse.node_base):
        if isinstance(r, g.ad.reverse.node_base) and n.gradient._container != r._container:
            n.gradient = value_of(n.gradient) - value_of(r)
        else:
            n.gradient = sub(n.gradient, r)
    elif isinstance(r, g.ad.reverse.node_base):
        n.gradient -= value_of(r)
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
            if accumulate_compatible(lhs_otype, rhs_spintrace_otype):
                backward_spin_trace = True
                rhs_otype = rhs_spintrace_otype
            elif rhs_spintrace_otype.colortrace[2] is not None:
                rhs_trace_otype = rhs_spintrace_otype.colortrace[2]()
                if accumulate_compatible(lhs_otype, rhs_trace_otype):
                    backward_trace = True
                    rhs_otype = rhs_trace_otype
        if rhs_otype.colortrace[2] is not None:
            rhs_colortrace_otype = rhs_otype.colortrace[2]()
            if accumulate_compatible(lhs_otype, rhs_colortrace_otype):
                backward_color_trace = True
                rhs_otype = rhs_colortrace_otype

        if not accumulate_compatible(rhs_otype, lhs_otype):
            raise Exception(
                "Conversion incomplete:" + rhs_otype.__name__ + ":" + lhs_otype.__name__
            )

    # g.message("Need to modify to",v._container,"from",c,":",backward_sum, backward_trace, backward_spin_trace, backward_color_trace)
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

            if (backward_trace or backward_color_trace or backward_spin_trace) and not isinstance(
                gradient, g.ad.reverse.node_base
            ):
                gradient = g(gradient)

            accum(v, gradient)

            # print("Ran conversion with sum/tr",backward_sum,backward_trace,backward_spin_trace,backward_color_trace)

    return g.ad.reverse.node_base(
        _forward,
        _backward,
        (v,),
        _container=c,
        _tag="change to " + str(c) + " from " + str(v._container),
    )
