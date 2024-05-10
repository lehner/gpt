#
#    GPT - Grid Python Toolkit
#    Copyright (C) 2020  Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)
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
import gpt
import cgpt
import numpy


def _simple_map(operator, numpy_operator=None, extra_params={}):
    def _mat(first, second=None):
        if isinstance(first, list):
            return [_mat(x) for x in first]
        if isinstance(first, gpt.expr):
            first = gpt(first)
        if isinstance(second, gpt.expr):
            second = gpt(second)
        return first.__class__.foundation.component_simple_map(
            operator, numpy_operator, extra_params, first, second
        )

    return _mat


imag = _simple_map("imag", numpy.imag)
real = _simple_map("real", numpy.real)
abs = _simple_map("abs", numpy.abs)
sqrt = _simple_map("sqrt", numpy.sqrt)
exp = _simple_map("exp", numpy.exp)
log = _simple_map("log", numpy.log)
sin = _simple_map("sin", numpy.sin)
asin = _simple_map("asin", numpy.arcsin)
cos = _simple_map("cos", numpy.cos)
acos = _simple_map("acos", numpy.arccos)
tan = _simple_map("tan", numpy.tan)
atan = _simple_map("atan", numpy.arctan)
sinh = _simple_map("sinh", numpy.sinh)
asinh = _simple_map("asinh", numpy.arcsinh)
cosh = _simple_map("cosh", numpy.cosh)
acosh = _simple_map("acosh", numpy.arccosh)
tanh = _simple_map("tanh", numpy.tanh)
atanh = _simple_map("atanh", numpy.arctanh)
inv = _simple_map("pow", lambda x: numpy.power(x, -1), extra_params={"exponent": -1.0})


def pow(exponent):
    return _simple_map(
        "pow", lambda x: numpy.power(x, exponent), extra_params={"exponent": exponent}
    )


def relu(a=0.0):
    return _simple_map("relu", extra_params={"a": a})


def drelu(a=0.0):
    return _simple_map("drelu", extra_params={"a": a})


def mod(n):
    return _simple_map("mod", extra_params={"n": n})


def multiply(a, b):
    return a.__class__.foundation.component_multiply(a, b)
