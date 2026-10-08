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
from gpt.ad.reverse.primitive import primitive


def _same(x, **static):
    # the container of a componentwise map: that of its argument (a copy:
    # containers are mutable, see node.set_otype)
    return x.copy()


def _linear_jvp(op):
    # the tangent rule of a (real-)linear componentwise map: dz = op(dx)
    return lambda z, children, tangents, **static: [op(t, **static) for t in tangents[0]]


def _chain_jvp(derivative):
    # the tangent rule of a componentwise function: dz = f'(x) dx
    # (componentwise), derivative(x, **static) -> f'(x)
    def _jvp(z, children, tangents, **static):
        d = derivative(children[0], **static)
        return [multiply(d, t) for t in tangents[0]]

    return _jvp


def _multiply_jvp(z, children, tangents):
    # d(a b) = da b + a db, componentwise
    (a, b), (ta, tb) = children, tangents
    k = len(ta if ta is not None else tb)
    out = []
    for j in range(k):
        terms = [] if ta is None else [multiply(ta[j], b)]
        terms += [] if tb is None else [multiply(a, tb[j])]
        out.append(terms[0] if len(terms) == 1 else terms[0] + terms[1])
    return out




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
    _same,
    vjp=lambda i, flow, x: conj(flow),
    reads=((),),
    jvp=_linear_jvp(lambda t: conj(t)),
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
    lambda a, b: a.copy(),
    vjp=lambda i, flow, a, b: multiply(flow, conj(b)) if i == 0 else multiply(conj(a), flow),
    reads=((1,), (0,)),
    jvp=_multiply_jvp,
)


# the derivative of relu, piecewise constant: no flow
drelu = primitive(
    "drelu",
    lambda x, a: g.component.drelu(a)(x),
    _same,
    vjp=lambda i, flow, x, a: None,
    reads=((),),
    jvp=lambda z, children, tangents, a: [None] * len(tangents[0]),
)

# relu: the flow into x is drelu(x) * flow, componentwise (drelu is real)
_relu = primitive(
    "relu",
    lambda x, a: g.component.relu(a)(x),
    _same,
    vjp=lambda i, flow, x, a: multiply(flow, drelu(x, a=a)),
    reads=((0,),),
    jvp=_chain_jvp(lambda x, a: drelu(x, a=a)),
)


def relu(x, a=0.0):
    return _relu(x, a=a)


# sin and cos: each other's derivatives; the flow into x is conj(cos x) *
# flow (sin) and -conj(sin x) * flow (cos), componentwise
sin = primitive(
    "sin",
    lambda x: g.component.sin(x),
    _same,
    vjp=lambda i, flow, x: multiply(flow, conj(cos(x))),
    reads=((0,),),
    jvp=_chain_jvp(lambda x: cos(x)),
)

cos = primitive(
    "cos",
    lambda x: g.component.cos(x),
    _same,
    vjp=lambda i, flow, x: -multiply(flow, conj(sin(x))),
    reads=((0,),),
    jvp=_chain_jvp(lambda x: -sin(x)),
)


def _plain_part(v, part):
    # the real or imaginary part in the container of v (a number gives a
    # float, whose container is that of numbers; arrays keep their dtype, so
    # the node container is unchanged)
    if g.util.is_num(v):
        return getattr(complex(v), part)
    if isinstance(v, np.ndarray):
        return getattr(v, part).astype(v.dtype)
    return getattr(g.component, part)(v)


# z = Re x depends on Re x only: with the gradient convention dL/dRe + i
# dL/dIm, the flow into x is the real part of the flow into z
real = primitive(
    "real",
    lambda x: _plain_part(x, "real"),
    _same,
    vjp=lambda i, flow, x: real(flow),
    reads=((),),
    jvp=_linear_jvp(lambda t: real(t)),
)


def _imag_vjp(i, flow, x):
    # z = Im x: dL/dIm x = dL/dRe z, so the flow into x is i times the real
    # part of the flow into z
    f = real(flow)
    return g(1j * f) if isinstance(f, g.lattice) else f * 1j


imag = primitive(
    "imag",
    lambda x: _plain_part(x, "imag"),
    _same,
    vjp=_imag_vjp,
    reads=((),),
    jvp=_linear_jvp(lambda t: imag(t)),
)
