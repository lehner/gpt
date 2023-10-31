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
from gpt.ad.reverse.util import accumulate


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


class node_base:
    def __init__(self, _forward, _backward=lambda z: None, _children=(), with_gradient=True):
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
        self.gradient = None

    # def __del__(self):
    # global gctr
    # gctr-=1
    # print(gctr)

    def zero_gradient(self):
        self.gradient = g(0 * self.value)

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
                accumulate(x.gradient, z.gradient * g.adj(y.value))
            if y.with_gradient:
                accumulate(y.gradient, g.adj(x.value) * z.gradient)

        return node_base(_forward, _backward, (x, y))

    def __rmul__(x, y):
        return node_base.__mul__(y, x)

    def __add__(x, y):
        def _forward():
            return x.value + y.value

        # not allowed to capture z, otherwise have reference loop!
        def _backward(z):
            if x.with_gradient:
                accumulate(x.gradient, z.gradient)
            if y.with_gradient:
                accumulate(y.gradient, z.gradient)

        return node_base(_forward, _backward, (x, y))

    def __sub__(x, y):
        def _forward():
            return x.value - y.value

        # not allowed to capture z, otherwise have reference loop!
        def _backward(z):
            if x.with_gradient:
                accumulate(x.gradient, z.gradient)
            if y.with_gradient:
                accumulate(y.gradient, -z.gradient)

        return node_base(_forward, _backward, (x, y))

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
                if eager:
                    n.value = g(n.value)

        if verbose_memory:
            g.message(
                f"Forward propagation through graph with {len(nodes)} nodes with maximum allocated fields: {max_fields_allocated}"
            )

    def backward(self, nodes, first_gradient):
        fields_allocated = len(nodes)  # .values
        max_fields_allocated = fields_allocated
        self.gradient = 1
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
                if n._forward is not None:
                    n.value = None
                    fields_allocated -= 1

        if verbose_memory:
            g.message(
                f"Backward propagation through graph with {len(nodes)} nodes with maximum allocated fields: {max_fields_allocated}"
            )

    def __call__(self, with_gradients=True):
        nodes = []
        forward_free = traverse(nodes, self)
        self.forward(nodes, free=forward_free if not with_gradients else None)
        if with_gradients:
            self.backward(nodes, first_gradient=forward_free)
        nodes = None
        return self.value

    def functional(self, *arguments):
        return node_differentiable_functional(self, arguments)



def node(x, with_gradient=True):
    return node_base(x, with_gradient=with_gradient)
