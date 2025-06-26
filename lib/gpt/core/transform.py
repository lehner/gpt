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
    if isinstance(first, gpt.expr):
        first = gpt.eval(first)
    if isinstance(first, gpt.expr):
        second = gpt.eval(second)
    return first.__class__.foundation.cshift(first, second, third, fourth)


def copy(first, second=None):
    return_list = isinstance(first, list)
    if second is not None:
        t = gpt.util.to_list(first)
        l = gpt.util.to_list(second)
    else:
        l = gpt.util.to_list(first)
        t = [x.new() for x in l]

    l[0].__class__.foundation.copy(t, l)
    if not return_list:
        return t[0]
    return t


def eval_list(a):
    return [gpt.eval(x) if isinstance(x, gpt.expr) else x for x in a]


def call_binary_aa_num(functional, a, b):
    return_list = (isinstance(a, list)) or (isinstance(b, list))
    a = gpt.util.to_list(a)
    b = gpt.util.to_list(b)
    res = functional(eval_list(a), eval_list(b))
    if return_list:
        return res
    return gpt.util.to_num(res[0, 0])


def call_unary_a_num(functional, a):
    return_list = isinstance(a, list)
    if not return_list:
        a = [a]
    a = eval_list(a)
    objects = {}
    indices = {}
    for n, x in enumerate(a):
        fnd = x.foundation
        if fnd not in objects:
            objects[fnd] = []
            indices[fnd] = []
        objects[fnd].append(x)
        indices[fnd].append(n)
    res = [None] * len(a)
    for fnd in objects:
        idx = indices[fnd]
        res_fnd = functional(objects[fnd])
        for i in range(len(idx)):
            res[idx[i]] = res_fnd[i]
    if return_list:
        return res
    return gpt.util.to_num(res[0])


def rank_inner_product(a, b, n_block=1, use_accelerator=True):
    return call_binary_aa_num(
        lambda la, lb: la[0].__class__.foundation.rank_inner_product(la, lb, n_block, use_accelerator), a, b
    )


def inner_product(a, b, n_block=1, use_accelerator=True):
    return call_binary_aa_num(
        lambda la, lb: la[0].__class__.foundation.inner_product(la, lb, n_block, use_accelerator), a, b
    )


def norm2(l):
    return call_unary_a_num(lambda la: la[0].__class__.foundation.norm2(la), l)


def object_rank_norm2(l):
    return call_unary_a_num(lambda la: la[0].__class__.foundation.object_rank_norm2(la), l)


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
    d = gpt.util.to_list(d)
    a = gpt.util.to_list(a)
    x = gpt.util.to_list(x)
    y = gpt.util.to_list(y)

    x = [gpt(v) for v in x]
    y = [gpt(v) for v in y]
    a = [complex(v) for v in a]

    for j in range(len(x)):
        for i in x[j].otype.v_idx:
            cgpt.lattice_axpy(d[j].v_obj[i], a[j], x[j].v_obj[i], y[j].v_obj[i])

    cgpt.accelerator_barrier()


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
    return src.__class__.foundation.identity(src)


def infinitesimal_to_cartesian(src, dsrc):
    if gpt.util.is_num(src):
        return dsrc
    return dsrc.__class__.foundation.infinitesimal_to_cartesian(src, dsrc)


def project(src, method):
    otype = src.otype
    otype.project(src, method)
    src.otype = otype
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
