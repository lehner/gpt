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
from gpt.ad.reverse import node


def inner_product(x, y):
    def _forward():
        return g.inner_product(x.value, y.value)

    # not allowed to capture z, otherwise have reference loop!
    def _backward(z):
        if x.with_gradient:
            x.gradient += y.value * g.adj(z.gradient)
        if y.with_gradient:
            y.gradient += x.value * g.adj(z.gradient)

    return node(_forward, _backward, (x, y))


def norm2(x):
    return inner_product(x, x)


def relu(x, a=0.0):
    def _forward():
        return g.component.relu(a)(x.value)

    # not allowed to capture z, otherwise have reference loop!
    def _backward(z):
        if x.with_gradient:
            active = g.component.drelu(a)(x.value)
            x.gradient += g.component.multiply(active, z.gradient)

    return node(_forward, _backward, (x,))
