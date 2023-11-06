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
from gpt.ad.reverse.util import accumulate_gradient


def inner_product(x, y):
    def _forward():
        return g.inner_product(x.value, y.value)

    # not allowed to capture z, otherwise have reference loop!
    def _backward(z):
        if x.with_gradient:
            accumulate_gradient(x, y.value * g.adj(z.gradient))
        if y.with_gradient:
            accumulate_gradient(y, x.value * g.adj(z.gradient))

    return g.ad.reverse.node_base(_forward, _backward, (x, y))


def norm2(x):
    assert len(x) == 1
    return [inner_product(x[0], x[0])]


def cshift(x, direction, displacement, none):
    assert none is None

    def _forward():
        return g.cshift(x.value, direction, displacement)

    # not allowed to capture z, otherwise have reference loop!
    def _backward(z):
        if x.with_gradient:
            accumulate_gradient(x, g.cshift(z.gradient, direction, -displacement))

    return g.ad.reverse.node_base(_forward, _backward, (x,))


def component_simple_map(operator, numpy_operator, extra_params, first, second):
    if operator == "relu":
        assert second is None
        return g.ad.reverse.transform.relu(first, a=extra_params["a"])
    raise Exception(f"component-wise operator {operator} not implemented in rev-AD")
