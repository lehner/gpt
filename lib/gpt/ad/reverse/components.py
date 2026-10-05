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
# Node operations between arrays, site-constant fields and the components of
# vector and matrix fields, in mutually adjoint pairs (the backward of each is the other,
# so that they nest to any order):
#
#   broadcast_array(a, template)   array a -> the field equal to a at every site
#   sum_to_array(x, template)      field x -> the array sum_x x(x)
#   component(v, i, template)      field v -> the scalar field v_i
#   embed(s, i, template)          scalar field s -> the field s e_i
#
# template: a plain lattice of the type of the field; i: the index tuple of a
# component of a vector (i,) or matrix (i, j) field.  The scalar fields are
# of type ot_singlet (the type of traces).
#
import gpt as g
import numpy as np
from gpt.ad.reverse.node import node_op
from gpt.ad.reverse.util import container, get_container, is_node, value_of

_plans = {}


def _component_plan(field, i):
    # a cached copy plan from component i of a field (a vector or matrix
    # field) to a scalar field, one per (type, grid, index)
    key = (field.describe(), field.grid.describe(), tuple(i))
    if key not in _plans:
        scalar = g.lattice(field.grid, g.ot_singlet())
        pos = g.coordinates(field)
        plan = g.copy_plan(scalar, field)
        plan.destination += scalar.view[pos]
        plan.source += field.view[(pos,) + tuple(i)]
        _plans[key] = plan()
    return _plans[key]


def broadcast_array(a, template):
    if not is_node(a):
        z = g.lattice(template)
        z[:] = a
        return z

    def forward():
        v = value_of(a)
        if is_node(v):
            return broadcast_array(v, template)
        z = g.lattice(template)
        z[:] = v
        return z

    return node_op(
        (a,),
        forward,
        (lambda z: (1, sum_to_array(z.gradient, template)),),
        get_container(template),
        "broadcast_array",
    )


def sum_to_array(x, template):
    if not is_node(x):
        return np.array(g.sum(g(x)).array, dtype=np.complex128)
    shape = tuple(template.otype.shape)

    def forward():
        v = value_of(x)
        return sum_to_array(v, template)

    return node_op(
        (x,),
        forward,
        (lambda z: (1, broadcast_array(z.gradient, template)),),
        container(np.ndarray, shape, np.complex128),
        "sum_to_array",
    )


def component(v, i, template):
    if not is_node(v):
        v = g(v)
        s = g.lattice(v.grid, g.ot_singlet())
        _component_plan(v, i)(s, v)
        return s

    def forward():
        return component(value_of(v), i, template)

    scalar = g.lattice(template.grid, g.ot_singlet())
    return node_op(
        (v,),
        forward,
        (lambda z: (1, embed(z.gradient, i, template)),),
        get_container(scalar),
        "component",
    )


_units = {}


def _unit_field(template, i):
    # the constant field e_i (one per type, grid and index)
    key = (template.describe(), template.grid.describe(), tuple(i))
    if key not in _units:
        e = g.lattice(template)
        a = np.zeros(template.otype.shape, dtype=np.complex128)
        a[tuple(i)] = 1
        e[:] = a
        _units[key] = e
    return _units[key]


def embed(s, i, template):
    if not is_node(s):
        # s e_i, a scalar times a constant unit field
        return g(g(s) * _unit_field(template, i))

    def forward():
        return embed(value_of(s), i, template)

    return node_op(
        (s,),
        forward,
        (lambda z: (1, component(z.gradient, i, template)),),
        get_container(template),
        "embed",
    )
