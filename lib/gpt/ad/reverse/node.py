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
from gpt.ad.reverse.util import accumulate_gradient, is_field
from gpt.ad.reverse import foundation
from gpt.core.foundation import base

verbose_memory = g.default.is_verbose("ad_memory")


def traverse(nodes, n, visited=None):
    # forward(children) = value
    # last usage
    root = visited is None
    if root:
        visited = set([])
    if n not in visited:
        visited.add(n)
        for c in n._children:
            traverse(nodes, c, visited)
        nodes.append(n)

    if root:
        last_need = {}
        for n in nodes:
            for x in n._children:
                last_need[x] = n

        forward_free = dict([(x, []) for x in nodes])
        for x in last_need:
            forward_free[last_need[x]].append(x)

        # forward free contains information for when we can
        # release the node.value for forward propagation
        return forward_free


class node_differentiable_functional(g.group.differentiable_functional):
    def __init__(self, node, arguments):
        self.node = node
        self.arguments = arguments

    def __call__(self, fields):
        assert len(fields) == len(self.arguments)
        for i in range(len(fields)):
            self.arguments[i].value @= fields[i]
        return self.node(with_gradients=False).real

    def gradient(self, fields, dfields):
        for a in self.arguments:
            a.with_gradient = False
        indices = [fields.index(df) for df in dfields]
        for i in indices:
            self.arguments[i].gradient = None
            self.arguments[i].with_gradient = True
        self.node()
        return [self.arguments[i].gradient for i in indices]


# gctr = 0


class node_base(base):
    foundation = foundation

    # TODO: deprecate infinitesimal_to_cartesian and make it default
    def __init__(
        self,
        _forward,
        _backward=lambda z: None,
        _children=(),
        with_gradient=True,
        infinitesimal_to_cartesian=True,
    ):
        # global gctr
        # gctr+=1
        if not callable(_forward):
            self._forward = None
            self.value = _forward
        else:
            self._forward = _forward
            self.value = None
        self._backward = _backward
        self._children = _children
        self.with_gradient = with_gradient
        self.infinitesimal_to_cartesian = infinitesimal_to_cartesian
        self.gradient = None

    # def __del__(self):
    # global gctr
    # gctr-=1
    # print(gctr)

    def zero_gradient(self):
        self.gradient = 0.0 * self.value
        if isinstance(self.gradient, g.expr):
            self.gradient = g(self.gradient)

    def __mul__(x, y):
        if not isinstance(x, node_base):
            x = node_base(x, with_gradient=False)

        if not isinstance(y, node_base):
            y = node_base(y, with_gradient=False)

        def _forward():
            return x.value * y.value

        # not allowed to capture z, otherwise have reference loop!
        def _backward(z):
            if x.with_gradient:
                accumulate_gradient(x, z.gradient * g.adj(y.value))
            if y.with_gradient:
                accumulate_gradient(y, g.adj(x.value) * z.gradient)

        return node_base(_forward, _backward, (x, y))

    def __rmul__(x, y):
        return node_base.__mul__(y, x)

    def __truediv__(self, other):
        return (1.0 / other) * self

    def __neg__(self):
        return (-1.0) * self

    def __getitem__(x, item):
        def getter(y):
            return y[item]

        def setter(y, z):
            y[item] = z

        return x.project(getter, setter)

    def project(x, getter, setter):
        def _forward():
            return getter(x.value)

        # not allowed to capture z, otherwise have reference loop!
        def _backward(z):
            if x.with_gradient:
                accumulate_gradient(x, z.gradient, getter, setter)

        return node_base(_forward, _backward, (x,))

    def __add__(x, y):
        if not isinstance(x, node_base):
            x = node_base(x, with_gradient=False)

        if not isinstance(y, node_base):
            y = node_base(y, with_gradient=False)

        def _forward():
            return x.value + y.value

        # not allowed to capture z, otherwise have reference loop!
        def _backward(z):
            if x.with_gradient:
                accumulate_gradient(x, z.gradient)
            if y.with_gradient:
                accumulate_gradient(y, z.gradient)

        return node_base(_forward, _backward, (x, y))

    def __sub__(x, y):
        if not isinstance(x, node_base):
            x = node_base(x, with_gradient=False)

        if not isinstance(y, node_base):
            y = node_base(y, with_gradient=False)

        def _forward():
            return x.value - y.value

        # not allowed to capture z, otherwise have reference loop!
        def _backward(z):
            if x.with_gradient:
                accumulate_gradient(x, z.gradient)
            if y.with_gradient:
                accumulate_gradient(y, -z.gradient)

        return node_base(_forward, _backward, (x, y))

    def __rsub__(x, y):
        return node_base.__sub__(y, x)

    def __radd__(x, y):
        return node_base.__add__(y, x)

    def forward(self, nodes, eager=True, free=None):
        max_fields_allocated = 0
        fields_allocated = 0
        for n in nodes:
            if n._forward is not None:
                n.value = n._forward()
                fields_allocated += 1
                max_fields_allocated = max(max_fields_allocated, fields_allocated)
                if free is not None:
                    free_n = free[n]
                    for m in free_n:
                        if m._forward is not None:
                            m.value = None
                            fields_allocated -= 1
                if eager and isinstance(n.value, g.expr):
                    n.value = g(n.value)

        if verbose_memory:
            g.message(
                f"Forward propagation through graph with {len(nodes)} nodes with maximum allocated fields: {max_fields_allocated}"
            )

    def backward(self, nodes, first_gradient, initial_gradient):
        fields_allocated = len(nodes)  # .values
        max_fields_allocated = fields_allocated
        if initial_gradient is None:
            if is_field(self.value):
                raise Exception(
                    "Expression evaluates to a field.  Gradient calculation is not unique."
                )
            if isinstance(self.value, complex) and abs(self.value.imag) > 1e-12 * abs(
                self.value.real
            ):
                raise Exception(
                    f"Expression does not evaluate to a real number ({self.value}).  Gradient calculation is not unique."
                )
            initial_gradient = 1.0
        self.zero_gradient()
        self.gradient += initial_gradient
        for n in reversed(nodes):
            first_gradient_n = first_gradient[n]
            for m in first_gradient_n:
                if m is not self:
                    m.zero_gradient()
                    fields_allocated += 1
                    max_fields_allocated = max(max_fields_allocated, fields_allocated)
            n._backward(n)
            if n._forward is not None:
                n.gradient = None
                fields_allocated -= 1
                if n is not self:
                    n.value = None
                    fields_allocated -= 1
            else:
                if n.with_gradient and n.infinitesimal_to_cartesian:
                    n.gradient = g.infinitesimal_to_cartesian(n.value, n.gradient)

        if verbose_memory:
            g.message(
                f"Backward propagation through graph with {len(nodes)} nodes with maximum allocated fields: {max_fields_allocated}"
            )

    # TODO: allow for lists of initial_gradients (could save forward runs at sake of more memory)
    def __call__(self, with_gradients=True, initial_gradient=None):
        nodes = []
        forward_free = traverse(nodes, self)
        self.forward(nodes, free=forward_free if not with_gradients else None)
        if with_gradients:
            self.backward(nodes, first_gradient=forward_free, initial_gradient=initial_gradient)
        nodes = None
        return self.value

    def functional(self, *arguments):
        return node_differentiable_functional(self, arguments)

    def get_grid(self):
        return self.value.grid

    def get_real(self):
        def getter(y):
            return y.real

        def setter(y, z):
            y @= z

        return self.project(getter, setter)

    grid = property(get_grid)
    real = property(get_real)


def node(x, with_gradient=True, infinitesimal_to_cartesian=True):
    return node_base(
        x, with_gradient=with_gradient, infinitesimal_to_cartesian=infinitesimal_to_cartesian
    )
