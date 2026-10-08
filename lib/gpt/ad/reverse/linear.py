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
# closed under their vjp (so that recorded passes reach any order):
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
from gpt.ad.reverse.primitive import primitive
from gpt.ad.reverse.util import constant, container, get_container, is_node


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


def _flows(f):
    # the flow into a list node as a list (missing entries None)
    return f if is_node(f) else list(f)


_stack = primitive(
    "stack",
    lambda *h: list(h),
    lambda *c: container(list, c[0], len(c)),
    vjp=lambda j, flow, *h: flow[j],
    reads=lambda n: ((),) * n,
)


def stack(h):
    # a list of fields (nodes or plain) as one list node
    if _is_list_node(h):
        return h
    if not any(is_node(x) for x in h):
        return constant(list(h))
    return _stack(*h)


def _matrix_vector_vjp(i, flow, W, h):
    if i == 0:
        return outer_sum(_flows(flow), h)
    return matrix_vector(dagger(W), _flows(flow))


matrix_vector = primitive(
    "matrix_vector",
    lambda W, h: _plain_matrix_vector(W, h),
    lambda W, h: container(list, h.tag[1], W.tag[1][0]),
    vjp=_matrix_vector_vjp,
    lift=(constant, stack),
    reads=((1,), (0,)),
)


def _outer_sum_vjp(i, flow, a, b):
    if i == 0:
        return matrix_vector(flow, b)
    return matrix_vector(dagger(flow), a)


outer_sum = primitive(
    "outer_sum",
    lambda a, b: _plain_outer_sum(a, b),
    lambda a, b: container(np.ndarray, (a.tag[2], b.tag[2]), np.complex128),
    vjp=_outer_sum_vjp,
    lift=(stack, stack),
    reads=((1,), (0,)),
)


dagger = primitive(
    "dagger",
    lambda W: np.conj(_array(W)).T.copy(),
    lambda W: container(np.ndarray, tuple(reversed(W.tag[1])), np.complex128),
    vjp=lambda i, flow, W: dagger(flow),
    reads=((),),
)


# element access a[index] of an indexable value (an array, a tensor, ...; a
# number for a single element) and its transpose, the zero of the value's
# type with one element set: a pair of primitives whose vjps are each other.
# A node's __getitem__ (other than of a list node) is element; a number
# parameter of g.ml is stored as a 0-d array and enters a function as
# element(box, ()).


def _plain_element(a, index, c):
    if c.tag[0] is np.ndarray:
        # (a number may stand for a 0-d array, e.g. a trial point of a check)
        a = np.asarray(a)
    v = a[index]
    if isinstance(v, np.ndarray):
        return v.item() if v.ndim == 0 else np.array(v)
    return v.item() if isinstance(v, np.generic) else v


def _plain_scatter(v, index, c):
    z = c.zero()
    if isinstance(z, np.ndarray) and not np.iscomplexobj(z):
        v = np.real(v)
    z[index] = v
    return z


_element = primitive(
    "element",
    _plain_element,
    lambda a, index, c: get_container(_plain_element(c.representative(), index, c)),
    vjp=lambda i, flow, a, index, c: scatter(flow, index, c),
    reads=((),),
)

_scatter = primitive(
    "scatter",
    _plain_scatter,
    lambda v, index, c: c.copy(),
    vjp=lambda i, flow, v, index, c: element(flow, index),
    reads=((),),
)


def element(a, index):
    # a[index] (plain or a node)
    return _element(a, index=index, c=get_container(a).copy())


def scatter(v, index, c):
    # the zero of the container c with v at index
    return _scatter(v, index=index, c=c)
