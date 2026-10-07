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


def joint_node(fields, forward, flows, z_container, name):
    """A node of fields (nodes; plain values are constants) whose backward
    computes the flows into all children at once, first order only (plain
    values and flows): forward(values) -> value; flows(values, z, needed) ->
    {child index: flow} for the indices in needed (the children with a
    gradient); z_container(containers) -> the container of the value."""
    op = primitive(
        name,
        lambda *v: forward(list(v)),
        lambda *c: z_container(c),
        joint_vjp=lambda z, needed, *v: flows(list(v), z, needed),
        order=1,
    )
    return op.node(*fields)


def functional_node(f, fields):
    """The value f(fields) of a g.group.differentiable_functional f as a node,
    so that it can be combined with other nodes (first order only: its
    backward is f.gradient, not a node graph).  fields: nodes (plain values
    are constants).  The flow into each field is f's cartesian gradient,
    converted to the infinitesimal convention of the flows and scaled by the
    flow into the result."""

    def flows(v, z, needed):
        # one gradient evaluation for all fields that need one
        seed = complex(z.gradient).real
        result = {}
        for j, gr in zip(needed, f.gradient(v, [v[j] for j in needed])):
            r = g.cartesian_to_infinitesimal(v[j], gr)
            result[j] = g(seed * r) if isinstance(r, g.lattice) else seed * r
        return result

    return joint_node(
        fields, lambda v: complex(f(v)), flows, lambda c: container(complex), "functional_node"
    )
