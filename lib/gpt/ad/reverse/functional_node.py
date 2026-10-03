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
from gpt.ad.reverse.node import node_base, node_op
from gpt.ad.reverse.util import container, is_node, value_of


def functional_node(f, fields):
    """The value f(fields) of a g.group.differentiable_functional f as a node,
    so that it can be combined with other nodes (first order only: its
    backward is f.gradient, not a node graph).  fields: nodes (plain values
    are constants).  The flow into each field is f's cartesian gradient,
    converted to the infinitesimal convention of the flows and scaled by the
    flow into the result."""
    children = tuple(x if is_node(x) else node_base(x, with_gradient=False) for x in fields)
    pending = {}

    def values():
        v = [value_of(c) for c in children]
        if any(is_node(x) for x in v):
            raise NotImplementedError("functional_node supports first derivatives only")
        return v

    def flow(i):
        def _backward(z):
            if is_node(z.gradient):
                raise NotImplementedError("functional_node supports first derivatives only")
            if not pending:
                # one gradient evaluation for all fields that need one
                v = values()
                needed = [j for j, c in enumerate(children) if c.with_gradient]
                for j, gr in zip(needed, f.gradient(v, [v[j] for j in needed])):
                    pending[j] = g.cartesian_to_infinitesimal(v[j], gr)
            seed = complex(z.gradient).real
            r = pending.pop(i)
            return (1, g(seed * r) if isinstance(r, g.lattice) else seed * r)

        return _backward

    return node_op(
        children,
        lambda: complex(f(values())),
        [flow(i) for i in range(len(children))],
        container(complex),
    )
