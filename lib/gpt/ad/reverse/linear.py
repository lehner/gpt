#
#    GPT - Grid Python Toolkit
#    Copyright (C) 2026  Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)
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
# Site-constant linear maps on lists of scalar fields, as node operations
# closed under their backward (so that they nest to any order):
#
#   stack(h)              a list of field nodes -> one list node
#   matrix_vector(W, h)   M x N array W, list h of N scalar fields -> the list
#                         of M fields y_i(x) = sum_j W_ij h_j(x)
#   outer_sum(a, b)       lists a (M fields), b (N fields) -> the M x N array
#                         sum_x a_i(x) conj(b_j(x))
#   dagger(W)             array W -> W^dag
#
# Flows: matrix_vector -> W: outer_sum(flow, h), h: matrix_vector(W^dag,
# flow); outer_sum -> a: matrix_vector(G, b), b: matrix_vector(G^dag, a);
# dagger -> dagger(flow); stack -> element j: flow[j].
#
# The plain kernels pack a list of fields into one contiguous accelerator
# buffer (site-major, the fields of a site contiguous: a sites x N matrix) and
# compute one gemm over all sites.
#
import gpt as g
import numpy as np
from gpt.ad.reverse.node import node_op
from gpt.ad.reverse.util import constant, container, is_node, value_of


def _array(x):
    return np.asarray(x, dtype=np.complex128)


def _plain_list(h, template=None):
    # a list of plain fields (missing flows, None, as zero fields)
    h = [None if x is None else g(x) for x in h]
    template = next(x for x in h if x is not None) if template is None else template
    result = []
    for x in h:
        if x is None:
            x = g.lattice(template)
            x[:] = 0
        result.append(x)
    return result


_kernels = {}


def _gemm(key, shape_a, shape_b, shape_c, op_a, op_b):
    # a cached gemm C = op(A) op(B) on fixed buffers (C zero-initialized:
    # with beta = 0 an implementation may still read C)
    if key not in _kernels:
        a = g.accelerator.buffer(shape=(1,) + shape_a, dtype=np.complex128)
        b = g.accelerator.buffer(shape=(1,) + shape_b, dtype=np.complex128)
        c = g.accelerator.buffer(np.zeros((1,) + shape_c, dtype=np.complex128))
        idx = np.array([0], dtype=np.int64)
        kernel = g.accelerator.kernel()
        va, vb = a[idx], b[idx]
        kernel.gemm(
            1.0, [va.H if op_a == "H" else va], [vb.H if op_b == "H" else vb], 0.0, [c[idx]]
        )
        _kernels[key] = (a, b, c, kernel)
    return _kernels[key]


def _pack_shape(fields):
    # the shape of the accelerator buffer of g.pack(fields)
    x = fields[0]
    return tuple(reversed(x.grid.ldimensions)) + (len(fields),) + tuple(x.otype.shape)


def _packed(fields, target):
    # the fields packed into the buffer target (shape (1, sites, n))
    view = g.accelerator.buffer(target).reshape(_pack_shape(fields))
    g.pack(fields).to_accelerator_buffer(target_buffer=view)


def _unpacked(fields, source):
    # the buffer source (shape (1, sites, n)) unpacked into the fields
    view = g.accelerator.buffer(source).reshape(_pack_shape(fields))
    g.pack(fields).from_accelerator_buffer(view)
    return fields


def _sites(field):
    return int(np.prod(field.grid.ldimensions))


def _plain_matrix_vector(W, h):
    W = _array(W)
    h = _plain_list(h)
    m, n = W.shape
    assert len(h) == n
    sites = _sites(h[0])
    key = ("matrix_vector", m, n, h[0].describe(), h[0].grid.describe())
    # Y (sites x m) = H (sites x n) W^T (n x m)
    a, b, c, kernel = _gemm(key, (sites, n), (n, m), (sites, m), "N", "N")
    _packed(h, a)
    b.from_array(np.ascontiguousarray(W.T).reshape((1, n, m)))
    kernel()
    return _unpacked([g.lattice(h[0]) for _ in range(m)], c)


def _plain_outer_sum(a, b):
    a, b = _plain_list(a), _plain_list(b)
    m, n = len(a), len(b)
    sites = _sites(a[0])
    key = ("outer_sum", m, n, a[0].describe(), a[0].grid.describe())
    # S^T (n x m) = B^H (n x sites) A (sites x m)
    ba, bb, bc, kernel = _gemm(key, (sites, n), (sites, m), (n, m), "H", "N")
    _packed(b, ba)
    _packed(a, bb)
    kernel()
    S = np.ascontiguousarray(np.asarray(bc.to_array())[0].T, dtype=np.complex128)
    return a[0].grid.globalsum(S)


def _is_list_node(x):
    return is_node(x) and x._container.tag[0] is list


def _flows(z):
    # the flow into a list node as a list (missing entries None)
    f = z.gradient
    if is_node(f):
        return f
    return list(f)


def stack(h):
    # a list of fields (nodes or plain) as one list node
    if _is_list_node(h):
        return h
    if not any(is_node(x) for x in h):
        return constant(list(h))
    h = [constant(x) for x in h]

    def forward():
        v = [value_of(x) for x in h]
        return stack(v) if any(is_node(x) for x in v) else v

    def backward(j):
        def _backward(z):
            fj = z.gradient[j]
            if fj is None:
                return None
            return (1, fj)

        return _backward

    return node_op(
        tuple(h),
        forward,
        tuple(backward(j) for j in range(len(h))),
        container(list, h[0]._container, len(h)),
        "stack",
    )


def dagger(W):
    if not is_node(W):
        return np.conj(_array(W)).T.copy()

    def forward():
        return dagger(value_of(W))

    m, n = W._container.tag[1]
    return node_op(
        (W,),
        forward,
        (lambda z: (1, dagger(z.gradient)),),
        container(np.ndarray, (n, m), np.complex128),
        "dagger",
        reads=((),),
    )


def matrix_vector(W, h):
    if not is_node(W) and not is_node(h) and not any(is_node(x) for x in h):
        return _plain_matrix_vector(W, h)
    W, h = constant(W), stack(h)
    m = W._container.tag[1][0]

    def forward():
        return matrix_vector(value_of(W), value_of(h))

    return node_op(
        (W, h),
        forward,
        (
            lambda z: (1, outer_sum(_flows(z), value_of(h))),
            lambda z: (1, matrix_vector(dagger(value_of(W)), _flows(z))),
        ),
        container(list, h._container.tag[1], m),
        "matrix_vector",
    )


def outer_sum(a, b):
    plain = lambda x: not is_node(x) and not any(is_node(y) for y in x)
    if plain(a) and plain(b):
        return _plain_outer_sum(a, b)
    a, b = stack(a), stack(b)
    m, n = a._container.tag[2], b._container.tag[2]

    def forward():
        return outer_sum(value_of(a), value_of(b))

    return node_op(
        (a, b),
        forward,
        (
            lambda z: (1, matrix_vector(z.gradient, value_of(b))),
            lambda z: (1, matrix_vector(dagger(z.gradient), value_of(a))),
        ),
        container(np.ndarray, (m, n), np.complex128),
        "outer_sum",
    )
