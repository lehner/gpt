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
import numpy as np
from gpt.ad.reverse.util import container, get_unary_container, get_container
import gpt.ad.reverse.foundation.matrix


def inner_product(x, y, n_block, use_accelerator):
    assert len(x) == 1 and len(y) == 1 and n_block == 1
    x = x[0]
    y = y[0]

    def _forward():
        return g.inner_product(x.value, y.value, n_block, use_accelerator)

    # not allowed to capture z, otherwise have reference loop!
    def _backward(z):
        if x.with_gradient:  # z = adj(x) y   ->    x z = y  -> x = y adj(z)
            x.gradient += y.value * g.adj(z.gradient)  # y.value * g.adj(z.gradient)
        if y.with_gradient:  # z = adj(x) y   ->    y = x z
            y.gradient += x.value * z.gradient

    return {
        (0, 0): g.ad.reverse.node_base(
            _forward, _backward, (x, y), _container=container(complex), _tag="inner_product"
        )
    }


def norm2(x):
    assert len(x) == 1
    return [g.inner_product(x, x)[0, 0]]


def cshift(x, direction, displacement, none):
    assert none is None

    def _forward():
        return g.cshift(x.value, direction, displacement)

    # not allowed to capture z, otherwise have reference loop!
    def _backward(z):
        if x.with_gradient:
            x.gradient += g.cshift(z.gradient, direction, -displacement)

    return g.ad.reverse.node_base(
        _forward,
        _backward,
        (x,),
        _container=x._container,
        _tag="cshift(" + str(direction) + ", " + str(displacement) + ")",
    )


def adj(x):
    def _forward():
        return g.adj(x.value)

    # not allowed to capture z, otherwise have reference loop!
    def _backward(z):
        if x.with_gradient:
            x.gradient += g.adj(z.gradient)

    return g.ad.reverse.node_base(_forward, _backward, (x,), _container=x._container, _tag="adj")


def trace(x, t):
    def _forward():
        return g.trace(x.value, t)

    # not allowed to capture z, otherwise have reference loop!
    def _backward(z):
        if x.with_gradient:
            x.gradient += g.identity(x.value) * z.gradient

    z_container = get_unary_container(x._container, lambda v: g.trace(v, t))

    return g.ad.reverse.node_base(_forward, _backward, (x,), _container=z_container)


def sum(x):
    def _forward():
        return g.sum(x.value)

    # not allowed to capture z, otherwise have reference loop!
    def _backward(z):
        if x.with_gradient:
            x.gradient += g.identity(x.value) * z.gradient

    return g.ad.reverse.node_base(
        _forward, _backward, (x,), _container=x._container.lattice_to_tensor()
    )


def component_simple_map(operator, numpy_operator, extra_params, first, second):
    if operator == "relu":
        assert second is None
        return g.ad.reverse.transform.relu(first, a=extra_params["a"])
    elif operator == "sin":
        assert second is None
        return g.ad.reverse.transform.sin(first)
    elif operator == "cos":
        assert second is None
        return g.ad.reverse.transform.cos(first)
    raise Exception(f"component-wise operator {operator} not implemented in rev-AD")


def infinitesimal_to_cartesian(src, dsrc):
    if gpt.util.is_num(src.value) or isinstance(src.value, np.ndarray):
        return dsrc
    return src.value.otype.infinitesimal_to_cartesian(src, dsrc)


def identity(x):
    def _forward():
        return g.identity(x.value)

    # not allowed to capture z, otherwise have reference loop!
    def _backward(z):
        pass

    return g.ad.reverse.node_base(
        _forward,
        _backward,
        (x,),
        _container=x._container,
        _tag="identity(" + str(x._container) + ")",
    )


def astype(x, y):
    def _forward():
        return g.astype(g(x.value), y)

    # not allowed to capture z, otherwise have reference loop!
    def _backward(z):
        if x.with_gradient:
            x.gradient += z.gradient

    z_container = x._container.copy()
    z_container.set_otype(y)

    return g.ad.reverse.node_base(
        _forward,
        _backward,
        (x,),
        _container=z_container,
        _tag="astype(" + str(x._container) + "," + str(y) + ")",
    )


def cshift_plan_add(self, fields, displacements):
    indices = {}
    for d in displacements:
        indices[d] = self.index
        self.index += 1
    self.indices.append(indices)
    return indices


def cshift_plan_execute(self):
    def _executer(first, second=None):
        assert second is None
        ret = []
        for i, displacements in enumerate(self.displacements):
            for d in displacements:
                ret.append(first[i])
                for dir, disp in enumerate(d):
                    if disp != 0:
                        ret[-1] = g.cshift(ret[-1], dir, disp)
        return ret

    return _executer
