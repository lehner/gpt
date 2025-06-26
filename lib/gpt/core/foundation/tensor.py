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


def rank_inner_product(a, b, n_block, use_accelerator):
    assert n_block == 1
    return numpy.array([[gpt.trace(gpt.adj(x) * y) for y in b] for x in a], dtype=numpy.complex128)


def inner_product(a, b, n_block, use_accelerator):
    assert n_block == 1
    return rank_inner_product(a, b, n_block, use_accelerator)


def norm2(a):
    return inner_product(a, a, len(a), True).real.reshape(len(a)).astype(numpy.float64)


def trace(a, t):
    return a.trace(t)


def component_simple_map(operator, numpy_operator, extra_params, first, second):
    assert second is None
    assert numpy_operator is not None
    res = first.new()
    res.array = numpy_operator(first.array)
    return res


def identity(t):
    e = gpt.tensor(t.otype)
    if len(e.array.shape) == 2:
        e.array = numpy.eye(dtype=e.array.dtype, N=e.array.shape[0])
    elif len(e.array.shape) == 4:
        n1 = e.array.shape[0]
        n2 = e.array.shape[2]
        for i in range(n1):
            for j in range(n1):
                if i == j:
                    e.array[i, j] = numpy.eye(dtype=e.array.dtype, N=n2)
                else:
                    e.array[i, j] = numpy.zeros(dtype=e.array.dtype, shape=(n2, n2))
    else:
        raise Exception(f"Unknown shape of tensor.identity {e.array.shape}")
    return e


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


def copy(a, b):
    for i in range(len(a)):
        a[i].array[:] = b[i].array[:]
