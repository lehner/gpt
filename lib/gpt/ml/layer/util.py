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
#
# Helpers shared by the layers.
#
import gpt as g
import numpy as np
from gpt.ad.reverse.util import is_node


def unit_scalar(template):
    # the unit scalar field, of the type of traces of the template (the
    # scalar type of coefficient and invariant fields)
    one = g(g.trace(g.identity(template)))
    one[:] = 1
    return one


def embed(c, template):
    # the field c(x) 1 of a number or complex scalar field c (plain or a
    # node; the backward is the trace), of the type of the template: a
    # coefficient times the unit matrix, or a global number as a field
    one = g.identity_constant(template)
    if is_node(c):
        return c * one
    return g(c * one)


def standardization(q):
    # the mean and inverse standard deviation of each invariant over the
    # sites of all samples (q: per sample, a list of real scalar fields), for
    # standardized invariants (q - mean) inv_std; 1 for an invariant that is
    # constant (it is only shifted)
    mean, inv_std = [], []
    for k in range(len(q[0])):
        n = sum(x[k].grid.gsites for x in q)
        m = sum(g.sum(x[k]).real for x in q) / n
        m2 = sum(g.sum(g(x[k] * x[k])).real for x in q) / n
        mean.append(m)
        var = m2 - m**2
        inv_std.append(1.0 / np.sqrt(var) if var > 1e-20 * (1.0 + m**2) else 1.0)
    return mean, inv_std
