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
import gpt
import numpy


def rank_inner_product(a, b, use_accelerator):
    return numpy.array([[gpt.trace(gpt.adj(x) * y) for y in b] for x in a], dtype=numpy.complex128)


def inner_product(a, b, use_accelerator):
    return rank_inner_product(a, b, use_accelerator)


def norm2(a):
    res = inner_product(a, a, True).real
    ip = numpy.ndarray(dtype=numpy.float64, shape=(len(a),))
    for i in range(len(a)):
        ip[i] = res[i, i]
    return ip


def trace(a, t):
    return a.trace(t)


def component_simple_map(operator, numpy_operator, extra_params, first, second):
    assert second is None
    assert numpy_operator is not None
    res = first.new()
    res.array = numpy_operator(first.array)
    return res


def adj(l):
    if l.transposable():
        return l.adj()
    return gpt.adj(gpt.expr(l))


def infinitesimal_to_cartesian(src, dsrc):
    return dsrc.otype.infinitesimal_to_cartesian(src, dsrc)


def component_multiply(a, b):
    res = a.new()
    res.array = numpy.multiply(a.array, b.array)
    return res
