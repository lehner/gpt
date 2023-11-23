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


def object_rank_norm2(l):
    return rank_inner_product(l, l, True).real


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


def identity(src):
    eye = gpt.lattice(src)
    # identity only works for matrix types
    n2 = len(eye.v_obj)
    n = int(n2**0.5)
    assert n * n == n2
    for i in range(n):
        for j in range(n):
            if i == j:
                cgpt.lattice_set_to_identity(eye.v_obj[i * n + j])
            else:
                cgpt.lattice_set_to_number(eye.v_obj[i * n + j], 0.0)
    return eye


def infinitesimal_to_cartesian(src, dsrc):
    return dsrc.otype.infinitesimal_to_cartesian(src, dsrc)


def group_inner_product(left, right):
    # inner product over group's real vector space
    left_type = left.otype
    return left_type.inner_product(left, right)


def copy(dst, src):
    for j in range(len(dst)):
        for i in dst[j].otype.v_idx:
            cgpt.copy(dst[j].v_obj[i], src[j].v_obj[i])


def convert(first, second):
    # following should go in foundation
    if second in [gpt.single, gpt.double, gpt.double_quadruple]:
        # if first is no list, evaluate
        src = gpt.eval(first)
        dst_grid = src.grid.converted(second)
        return convert(gpt.lattice(dst_grid, src.otype), src)

    elif isinstance(second, gpt.ot_base):
        # if first is no list, evaluate
        src = gpt.eval(first)
        if src.otype.__name__ == second.__name__:
            return gpt.copy(src)
        return convert(gpt.lattice(src.grid, second), src)

    elif isinstance(first, gpt.lattice):
        # second may be expression
        second = gpt.eval(second)

        # if otypes differ, attempt otype conversion first
        if first.otype.__name__ != second.otype.__name__:
            assert first.otype.__name__ in second.otype.ctab
            tmp = gpt.lattice(first)
            second.otype.ctab[first.otype.__name__](tmp, second)
            second = tmp
            assert first.otype.__name__ == second.otype.__name__

        # convert precision if needed
        if first.grid == second.grid:
            gpt.copy(first, second)

        else:
            assert len(first.otype.v_idx) == len(second.otype.v_idx)
            for i in first.otype.v_idx:
                cgpt.convert(first.v_obj[i], second.v_obj[i])
            first.checkerboard(second.checkerboard())

        return first

    else:
        assert 0


def matrix_det(A):
    r = gpt.complex(A.grid)
    to_list = gpt.util.to_list
    cgpt.determinant(r.v_obj[0], to_list(A))
    return r


def component_multiply(a, b):
    a = gpt(a)
    b = gpt(b)
    assert a.otype.__name__ == b.otype.__name__
    res = gpt.lattice(a)
    params = {"operator": "*"}
    n = len(res.v_obj)
    assert n == len(a.v_obj)
    assert n == len(b.v_obj)
    for i in range(n):
        cgpt.binary(res.v_obj[i], a.v_obj[i], b.v_obj[i], params)
    return res
