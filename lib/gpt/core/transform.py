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
import cgpt, gpt, numpy


def cshift(first, second, third, fourth=None):
    if isinstance(first, gpt.lattice) and isinstance(second, gpt.lattice) and fourth is not None:
        t = first
        l = gpt.eval(second)
        d = third
        o = fourth
    else:
        l = gpt.eval(first)
        d = second
        o = third
        t = gpt.lattice(l)

    for i in t.otype.v_idx:
        cgpt.cshift(t.v_obj[i], l.v_obj[i], d, o)
    return t


def copy(first, second=None):
    if second is not None:
        t = first
        l = second

    else:
        l = first
        if isinstance(l, list):
            t = [gpt.lattice(x) for x in l]
        else:
            t = gpt.lattice(l)

    if isinstance(l, gpt.lattice):
        for i in t.otype.v_idx:
            cgpt.copy(t.v_obj[i], l.v_obj[i])
    else:
        for j in range(len(l)):
            for i in t[j].otype.v_idx:
                cgpt.copy(t[j].v_obj[i], l[j].v_obj[i])

    return t


def rank_inner_product(a, b, use_accelerator=True):
    return_list = (isinstance(a, list)) or (isinstance(b, list))
    a = gpt.util.to_list(a)
    b = gpt.util.to_list(b)
    if isinstance(a[0], gpt.tensor) and isinstance(b[0], gpt.tensor):
        res = numpy.array([[gpt.adj(x) * y for y in b] for x in a], dtype=numpy.complex128)
    else:
        a = [gpt.eval(x) for x in a]
        b = [gpt.eval(x) for x in b]
        otype = a[0].otype
        assert len(otype.v_idx) == len(b[0].otype.v_idx)
        res = cgpt.lattice_rank_inner_product(a, b, use_accelerator)
    if return_list:
        return res
    return gpt.util.to_num(res[0, 0])


def inner_product(a, b):
    if isinstance(a, gpt.tensor):
        return gpt.trace(gpt.adj(a) * b)
    grid = gpt.util.to_list(a)[0].grid
    return grid.globalsum(rank_inner_product(a, b))


def norm2(l):
    l = gpt.eval(l)
    return_list = isinstance(l, list)
    l = gpt.util.to_list(l)
    l_lattices = [(i, l[i]) for i in range(len(l)) if isinstance(l[i], gpt.lattice)]
    l_tensors = [(i, l[i]) for i in range(len(l)) if isinstance(l[i], gpt.tensor)]
    ip_l = (
        l_lattices[0][1]
        .grid.globalsum(
            numpy.array([rank_inner_product(x, x) for i, x in l_lattices], dtype=numpy.complex128)
        )
        .real
    )
    ip_t = [x.norm2() for i, x in l_tensors]
    ip = numpy.ndarray(dtype=numpy.complex128, shape=(len(l),))
    for i, j in enumerate(l_lattices):
        ip[j[0]] = ip_l[i]
    for i, j in enumerate(l_tensors):
        ip[j[0]] = ip_t[i]
    if return_list:
        return ip
    return gpt.util.to_num(ip[0])


def inner_product_norm2(a, b):
    if isinstance(a, gpt.tensor) and isinstance(b, gpt.tensor):
        return gpt.adj(a) * b, a.norm2()
    a = gpt.eval(a)
    b = gpt.eval(b)
    assert len(a.otype.v_idx) == len(b.otype.v_idx)
    r = [cgpt.lattice_inner_product_norm2(a.v_obj[i], b.v_obj[i]) for i in a.otype.v_idx]
    return (
        sum([x[0] for x in r]),
        sum([x[1] for x in r]),
    )  # todo, make local version of this too


def axpy(d, a, x, y):
    x = gpt.eval(x)
    y = gpt.eval(y)
    a = complex(a)
    assert len(y.otype.v_idx) == len(x.otype.v_idx)
    assert len(d.otype.v_idx) == len(x.otype.v_idx)
    for i in x.otype.v_idx:
        cgpt.lattice_axpy(d.v_obj[i], a, x.v_obj[i], y.v_obj[i])


def axpy_norm2(d, a, x, y):
    axpy(d, a, x, y)
    return norm2(d)


def fields_to_tensors(src, functor):
    return_list = isinstance(src, list)
    src = gpt.util.to_list(gpt.eval(src))

    # check for consistent otype
    assert all([src[0].otype.__name__ == obj.otype.__name__ for obj in src])

    result = functor(src)

    if return_list:
        return [[gpt.util.value_to_tensor(v, src[0].otype) for v in res] for res in result]
    return [gpt.util.value_to_tensor(v, src[0].otype) for v in result[0]]


def slice(src, dim):
    return fields_to_tensors(src, lambda s: s[0].grid.globalsum(cgpt.lattice_rank_slice(s, dim)))


def indexed_sum(fields, index, length):
    index_obj = index.v_obj[0]
    return fields_to_tensors(
        fields, lambda s: s[0].grid.globalsum(cgpt.lattice_rank_indexed_sum(s, index_obj, length))
    )


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


def project(src, method):
    src.otype.project(src, method)
    return src


def where(first, second, third, fourth=None):
    if fourth is None:
        question = first
        yes = second
        no = third
        answer = None
    else:
        question = second
        yes = third
        no = fourth
        answer = first

    question = gpt.eval(question)
    yes = gpt.eval(yes)
    no = gpt.eval(no)
    if answer is None:
        answer = gpt.lattice(yes)

    assert len(question.v_obj) == 1
    assert len(yes.v_obj) == len(no.v_obj)
    assert len(answer.v_obj) == len(yes.v_obj)

    params = {"operator": "?:"}

    for a, y, n in zip(answer.v_obj, yes.v_obj, no.v_obj):
        cgpt.ternary(a, question.v_obj[0], y, n, params)

    return answer


def scale_per_coordinate(d, s, a, dim):
    s = gpt.eval(s)
    assert len(d.otype.v_idx) == len(s.otype.v_idx)
    for i in d.otype.v_idx:
        cgpt.lattice_scale_per_coordinate(d.v_obj[i], s.v_obj[i], a, dim)
