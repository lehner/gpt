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
from gpt.ad.reverse import node_op
from gpt.ad.reverse.util import product, value_of


def relu(x, a=0.0):
    return node_op(
        (x,),
        lambda: g.component.relu(a)(value_of(x)),
        (lambda z: (1, g.component.multiply(g.component.drelu(a)(value_of(x)), z.gradient)),),
        x._container,
    )


def sin(x):
    return node_op(
        (x,),
        lambda: g.component.sin(value_of(x)),
        # conjugate-linear convention (matches __mul__): adj(cos(x)) * flow.
        # g.adj(lattice) is a symbolic expr, so use product (which handles
        # expr via the symbolic path, as __mul__ does) rather than
        # g.component.multiply (which requires a concrete lattice operand).
        (lambda z: (1, product(z.gradient, g.adj(g.component.cos(value_of(x))))),),
        x._container,
    )


def cos(x):
    return node_op(
        (x,),
        lambda: g.component.cos(value_of(x)),
        # conjugate-linear convention (matches __mul__): adj(-sin(x)) * flow
        (lambda z: (-1, product(z.gradient, g.adj(g.component.sin(value_of(x))))),),
        x._container,
    )
