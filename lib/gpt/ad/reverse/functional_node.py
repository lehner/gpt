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
import gpt as g
from gpt.ad.reverse.primitive import primitive
from gpt.ad.reverse.util import container


def functional_node(f, fields):
    """The value f(fields) of a g.group.differentiable_functional f as a node,
    so that it can be combined with other nodes (first order only: its
    backward is f.gradient, not a node graph).  fields: nodes (plain values
    are constants).  The flow into each field is f's cartesian gradient,
    converted to the infinitesimal convention of the flows and scaled by the
    flow into the result."""

    def joint_vjp(z, needed, *v):
        # one gradient evaluation for all fields that need one
        v = list(v)
        seed = complex(z.gradient).real
        result = {}
        for j, gr in zip(needed, f.gradient(v, [v[j] for j in needed])):
            r = g.cartesian_to_infinitesimal(v[j], gr)
            result[j] = g(seed * r) if isinstance(r, g.lattice) else seed * r
        return result

    op = primitive(
        "functional_node",
        lambda *v: complex(f(list(v))),
        lambda *c: container(complex),
        joint_vjp=joint_vjp,
        order=1,
    )
    return op.node(*fields)
