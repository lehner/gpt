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


def traverse(nodes, visited, n):
    if id(n) not in visited:
        visited.add(id(n))
        for c in n._children:
            traverse(nodes, visited, c)
        nodes.append(n)


class node:
    def __init__(self, _forward, _backward=lambda z: None, _children=(), with_gradient=True):
        if not callable(_forward):
            self._forward = lambda: _forward
        else:
            self._forward = _forward
        self._backward = _backward
        self._children = _children
        self.with_gradient = with_gradient
        self.value = None
        self.gradient = None

    def zero_gradient(self):
        self.gradient = g(0 * self.value)

    def __mul__(x, y):
        if not isinstance(x, node):
            x = node(x, with_gradient=False)

        if not isinstance(y, node):
            y = node(y, with_gradient=False)

        def _forward():
            return x.value * y.value

        # not allowed to capture z, otherwise have reference loop!
        def _backward(z):
            if x.with_gradient:
                x.gradient += z.gradient * g.adj(y.value)
            if y.with_gradient:
                y.gradient += g.adj(x.value) * z.gradient

        return node(_forward, _backward, (x, y))

    def __rmul__(x, y):
        return node.__mul__(y, x)

    def __add__(x, y):
        def _forward():
            return x.value + y.value

        # not allowed to capture z, otherwise have reference loop!
        def _backward(z):
            if x.with_gradient:
                x.gradient += z.gradient
            if y.with_gradient:
                y.gradient += z.gradient

        return node(_forward, _backward, (x, y))

    def __sub__(x, y):
        def _forward():
            return x.value - y.value

        # not allowed to capture z, otherwise have reference loop!
        def _backward(z):
            if x.with_gradient:
                x.gradient += z.gradient
            if y.with_gradient:
                y.gradient -= z.gradient

        return node(_forward, _backward, (x, y))

    def forward(self, nodes, eager=True):
        for n in nodes:
            n.value = n._forward()
            if eager:
                n.value = g(n.value)

    def backward(self, nodes):
        for n in nodes:
            n.zero_gradient()
        self.gradient = 1
        for n in reversed(nodes):
            n._backward(n)

    def __call__(self, with_gradients=True):
        visited = set([])
        nodes = []
        traverse(nodes, visited, self)
        self.forward(nodes)
        if with_gradients:
            self.backward(nodes)
        nodes = None
        visited = None
        return self.value
