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
from gpt.ad.reverse.primitive import primitive
from gpt.ad.reverse.util import value_of


def relu(x, a=0.0):
    return node_op(
        (x,),
        lambda: g.component.relu(a)(value_of(x)),
        (lambda z: (1, g.component.multiply(g.component.drelu(a)(value_of(x)), z.gradient)),),
        x._container,
    )


def _plain_conj(v):
    if g.util.is_num(v):
        return complex(v).conjugate()
    if isinstance(v, np.ndarray):
        return np.conj(v)
    return g(g.conj(v))


# the componentwise complex conjugate (no transpose); with the gradient
# convention dL/dRe + i dL/dIm the flow into x is conj(flow)
conj = primitive(
    "conj",
    _plain_conj,
    lambda x: x,
    vjp=lambda i, flow, x: conj(flow),
    reads=((),),
)


def _plain_multiply(a, b):
    if g.util.is_num(a) or isinstance(a, np.ndarray):
        return a * b
    return g.lattice.foundation.component_multiply(g(a), g(b))


# the componentwise product (for scalar types the ordinary product): flows
# conj(b) * flow into a and conj(a) * flow into b, componentwise
multiply = primitive(
    "multiply",
    _plain_multiply,
    lambda a, b: a,
    vjp=lambda i, flow, a, b: multiply(flow, conj(b)) if i == 0 else multiply(conj(a), flow),
)


def component_multiply(a, b):
    # the componentwise product of plain values or nodes (a node operation
    # if either is a node)
    return multiply(a, b)


def sin(x):
    # the flow into x is conj(cos x) * flow, componentwise
    return node_op(
        (x,),
        lambda: g.component.sin(value_of(x)),
        (lambda z: (1, multiply(z.gradient, conj(g.component.cos(value_of(x))))),),
        x._container,
    )


def cos(x):
    # the flow into x is -conj(sin x) * flow, componentwise
    return node_op(
        (x,),
        lambda: g.component.cos(value_of(x)),
        (lambda z: (-1, multiply(z.gradient, conj(g.component.sin(value_of(x))))),),
        x._container,
    )


def _plain_part(v, part):
    # the real or imaginary part in the container of v (numbers stay complex
    # and arrays keep their dtype, so the node container is unchanged)
    if g.util.is_num(v):
        return complex(getattr(complex(v), part))
    if isinstance(v, np.ndarray):
        return getattr(v, part).astype(v.dtype)
    return getattr(g.component, part)(v)


# z = Re x depends on Re x only: with the gradient convention dL/dRe + i
# dL/dIm, the flow into x is the real part of the flow into z
real = primitive(
    "real",
    lambda x: _plain_part(x, "real"),
    lambda x: x,
    vjp=lambda i, flow, x: real(flow),
)


def _imag_vjp(i, flow, x):
    # z = Im x: dL/dIm x = dL/dRe z, so the flow into x is i times the real
    # part of the flow into z
    f = real(flow)
    return g(1j * f) if isinstance(f, g.lattice) else f * 1j


imag = primitive(
    "imag",
    lambda x: _plain_part(x, "imag"),
    lambda x: x,
    vjp=_imag_vjp,
)
