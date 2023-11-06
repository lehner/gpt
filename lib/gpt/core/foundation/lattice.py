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
import cgpt
import numpy


def rank_inner_product(a, b, use_accelerator):
    otype = a[0].otype
    assert len(otype.v_idx) == len(b[0].otype.v_idx)
    return cgpt.lattice_rank_inner_product(a, b, use_accelerator)


def inner_product(a, b, use_accelerator):
    return a[0].grid.globalsum(rank_inner_product(a, b, use_accelerator))


def norm2(l):
    return (
        l[0]
        .grid.globalsum(
            numpy.array(
                [rank_inner_product([x], [x], True)[0, 0] for x in l], dtype=numpy.complex128
            )
        )
        .real
    )


def cshift(first, second, third, fourth):
    if fourth is not None:
        l = second
        d = third
        o = fourth
        t = first
    else:
        l = first
        d = second
        o = third
        t = gpt.lattice(l)

    for i in t.otype.v_idx:
        cgpt.cshift(t.v_obj[i], l.v_obj[i], d, o)
    return t


def trace(l, t):
    return gpt.expr(l, t)


def component_simple_map(operator, numpy_operator, extra_params, first, second):
    if second is not None:
        dst = first
        src = second
    else:
        src = first
        dst = gpt.lattice(src)
    for i in dst.otype.v_idx:
        cgpt.unary(dst.v_obj[i], src.v_obj[i], {**{"operator": operator}, **extra_params})
    return dst


def adj(l):
    return gpt.adj(gpt.expr(l))


def rank_sum(l):
    val = [cgpt.lattice_rank_sum(x) for x in l.v_obj]
    vrank = len(val)
    if vrank == 1:
        val = val[0]
    else:
        vdim = len(l.otype.shape)
        if vdim == 1:
            val = numpy.concatenate(val)
        elif vdim == 2:
            n = int(vrank**0.5)
            assert n * n == vrank
            val = numpy.concatenate(
                [numpy.concatenate([val[i * n + j] for j in range(n)], axis=0) for i in range(n)],
                axis=1,
            )
        else:
            raise NotImplementedError()
    return gpt.util.value_to_tensor(val, l.otype)


def sum(l):
    return l.grid.globalsum(rank_sum(l))
