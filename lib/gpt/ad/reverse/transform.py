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
from gpt.ad.reverse import node_op
from gpt.ad.reverse.util import value_of, is_node, nodify


def relu(x, a=0.0):
    return node_op(
        (x,),
        lambda: g.component.relu(a)(value_of(x)),
        (lambda z: (1, g.component.multiply(g.component.drelu(a)(value_of(x)), z.gradient)),),
        x._container,
    )


def _conj(v):
    if is_node(v):
        return conj(v)
    if g.util.is_num(v):
        return complex(v).conjugate()
    if isinstance(v, np.ndarray):
        return np.conj(v)
    return g(g.conj(v))


def conj(x):
    # the componentwise complex conjugate (no transpose); with the gradient
    # convention dL/dRe + i dL/dIm the flow into x is conj(flow)
    return node_op(
        (x,),
        lambda: _conj(value_of(x)),
        (lambda z: (1, _conj(z.gradient)),),
        x._container,
        "conj",
        reads=((),),
    )


def component_multiply(a, b):
    # the componentwise product of plain values or nodes (a node operation
    # if either is a node)
    if is_node(a) or is_node(b):
        return multiply(a, b)
    if g.util.is_num(a) or isinstance(a, np.ndarray):
        return a * b
    return g.lattice.foundation.component_multiply(g(a), g(b))


def multiply(a, b):
    # the componentwise product (for scalar types the ordinary product):
    # flows conj(b) * flow into a and conj(a) * flow into b, componentwise
    a, b = nodify(a, b)
    return node_op(
        (a, b),
        lambda: component_multiply(value_of(a), value_of(b)),
        (
            lambda z: (1, component_multiply(z.gradient, _conj(value_of(b)))),
            lambda z: (1, component_multiply(_conj(value_of(a)), z.gradient)),
        ),
        a._container,
        "multiply",
    )


def sin(x):
    # the flow into x is conj(cos x) * flow, componentwise
    return node_op(
        (x,),
        lambda: g.component.sin(value_of(x)),
        (lambda z: (1, component_multiply(z.gradient, _conj(g.component.cos(value_of(x))))),),
        x._container,
    )


def cos(x):
    # the flow into x is -conj(sin x) * flow, componentwise
    return node_op(
        (x,),
        lambda: g.component.cos(value_of(x)),
        (lambda z: (-1, component_multiply(z.gradient, _conj(g.component.sin(value_of(x))))),),
        x._container,
    )


def _part(v, part):
    # the real or imaginary part in the container of v (numbers stay complex
    # and arrays keep their dtype, so the node container is unchanged); a
    # node value (a nested pass) gives the node operation one level down
    if is_node(v):
        return real(v) if part == "real" else imag(v)
    if g.util.is_num(v):
        return complex(getattr(complex(v), part))
    if isinstance(v, np.ndarray):
        return getattr(v, part).astype(v.dtype)
    return getattr(g.component, part)(v)


def _real_flow(f):
    # the real part of a flow, as a node at deeper nesting
    return real(f) if is_node(f) else _part(f, "real")


def real(x):
    # z = Re x depends on Re x only: with the gradient convention
    # dL/dRe + i dL/dIm, the flow into x is the real part of the flow into z
    return node_op(
        (x,),
        lambda: _part(value_of(x), "real"),
        (lambda z: (1, _real_flow(z.gradient)),),
        x._container,
    )


def imag(x):
    # z = Im x: dL/dIm x = dL/dRe z, so the flow into x is i times the real
    # part of the flow into z
    def _flow(z):
        f = _real_flow(z.gradient)
        return (1, g(1j * f) if isinstance(f, g.lattice) else f * 1j)

    return node_op(
        (x,),
        lambda: _part(value_of(x), "imag"),
        (_flow,),
        x._container,
    )
