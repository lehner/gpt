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
# compute one gemm over all sites.  Packing costs more than the gemm, so the
# plain backward of matrix_vector packs each list once: one joint vjp packs
# the flow once for both flows, the weight flow reads the h packed by the
# forward (a residual), and only the entries of h that carry a gradient get
# a flow (not, e.g., a constant bias field).
#
import gpt as g
import numpy as np
from gpt.ad.reverse.primitive import primitive, has_node
from gpt.ad.reverse import tangent
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


def _pack(fields):
    # the fields packed into a new buffer (1, sites, n) (allocating a buffer
    # is cheap, packing is not)
    buffer = g.accelerator.buffer(shape=(1, _sites(fields[0]), len(fields)), dtype=np.complex128)
    _packed(fields, buffer)
    return buffer


# the output buffers of the gemms, per shape, reused (zero-initialized at
# creation: with beta = 0 an implementation may still read the output)
_outputs = {}


def _gemm(a, b, shape_c, op_a="N", op_b="N"):
    # op(a) op(b) for buffers (1, rows, columns) (op: "N" or "H"), in the
    # output buffer of shape_c, which the next gemm of that shape overwrites
    # (a kernel per call: building one is cheap)
    if shape_c not in _outputs:
        _outputs[shape_c] = g.accelerator.buffer(np.zeros((1,) + shape_c, dtype=np.complex128))
    c = _outputs[shape_c]
    idx = np.array([0], dtype=np.int64)
    va, vb = a[idx], b[idx]
    kernel = g.accelerator.kernel()
    kernel.gemm(1.0, [va.H if op_a == "H" else va], [vb.H if op_b == "H" else vb], 0.0, [c[idx]])
    kernel()
    return c


def _fields_times(X, M, template):
    # the l fields of X M for packed fields X (1, sites, k) and a k x l array M
    k, l = M.shape
    b = g.accelerator.buffer(np.ascontiguousarray(M, dtype=np.complex128).reshape((1, k, l)))
    return _unpacked([g.lattice(template) for _ in range(l)], _gemm(X, b, (X.shape[1], l)))


def _outer(B, A, grid):
    # the M x N array sum_x a_i(x) conj(b_j(x)) of packed fields A (M), B (N):
    # its transpose is B^H A
    c = _gemm(B, A, (B.shape[2], A.shape[2]), "H", "N")
    S = np.ascontiguousarray(np.asarray(c.to_array())[0].T, dtype=np.complex128)
    return grid.globalsum(S)


def _fwd_matrix_vector(W, h):
    # Y (sites x m) = H (sites x n) W^T (n x m); the packed H is the residual
    # (the weight flow reads it)
    W, h = _array(W), _plain_list(h)
    assert len(h) == W.shape[1]
    H = _pack(h)
    return _fields_times(H, W.T, h[0]), H


def _plain_matrix_vector(W, h):
    return _fwd_matrix_vector(W, h)[0]


def _plain_outer_sum(a, b):
    a, b = _plain_list(a), _plain_list(b)
    return _outer(_pack(b), _pack(a), a[0].grid)


def _is_list_node(x):
    return is_node(x) and x._container.tag[0] is list


def _flows(f):
    # the flow into a list node as a list (missing entries None)
    return f if is_node(f) else list(f)


def _stack_jvp(z, children, tangents):
    # the stack of the tangents (zeros for constant entries)
    k = tangent.count(tangents)

    def entry(i, j):
        if tangents[i] is not None:
            return tangents[i][j]
        return constant(children[i]._container.zero())

    return [_stack(*[entry(i, j) for i in range(len(children))]) for j in range(k)]


_stack = primitive(
    "stack",
    lambda *h: list(h),
    lambda *c: container(list, c[0], len(c)),
    vjp=lambda j, flow, *h: flow[j],
    reads=lambda n: ((),) * n,
    jvp=_stack_jvp,
)


def stack(h):
    # a list of fields (nodes or plain) as one list node
    if _is_list_node(h):
        return h
    if not any(is_node(x) for x in h):
        return constant(list(h))
    return _stack(*h)


def _add_lists(x, y):
    # the sum of two list nodes (elementwise), or of two values
    if _is_list_node(x):
        return _stack(*[x[i] + y[i] for i in range(len(x))])
    return x + y


def _matrix_vector_vjp(z, needed, W, h, residual):
    # W: outer_sum(flow, h), h: matrix_vector(W^dag, flow)
    flow = z.gradient
    if has_node(flow) or is_node(W) or has_node(h):
        # recorded: the flows as primitives (nodes)
        flow = _flows(flow)
        result = {}
        if 0 in needed:
            result[0] = outer_sum(flow, h)
        if 1 in needed:
            result[1] = matrix_vector(dagger(W), flow)
        return result
    # plain: the flow packed once for both flows
    flow = _plain_list(flow)
    F = _pack(flow)
    result = {}
    if 0 in needed:
        # (no residual: the forward did not run in this pass)
        H = _pack(_plain_list(h)) if residual is None else residual
        result[0] = _outer(H, F, flow[0].grid)
    hn = z._children[1]
    n = hn._container.tag[2]
    cols = list(range(n))
    if hn._tag == "stack":
        # the flows of the entries of h that carry a gradient
        cols = [j for j, c in enumerate(hn._children) if c.with_gradient]
    if 1 in needed and cols:
        y = _fields_times(F, np.conj(_array(W)[:, cols]), flow[0])
        r = [None] * n
        for j, x in zip(cols, y):
            r[j] = x
        result[1] = r
    return result


matrix_vector = primitive(
    "matrix_vector",
    lambda W, h: _plain_matrix_vector(W, h),
    lambda W, h: container(list, h.tag[1], W.tag[1][0]),
    joint_vjp=_matrix_vector_vjp,
    fwd=lambda W, h: _fwd_matrix_vector(W, h),
    lift=(constant, stack),
    reads=((1,), (0,)),
    jvp=tangent.bilinear(lambda W, h: matrix_vector(W, h), _add_lists),
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
    jvp=tangent.bilinear(lambda a, b: outer_sum(a, b), _add_lists),
)


dagger = primitive(
    "dagger",
    lambda W: np.conj(_array(W)).T.copy(),
    lambda W: container(np.ndarray, tuple(reversed(W.tag[1])), np.complex128),
    vjp=lambda i, flow, W: dagger(flow),
    reads=((),),
    jvp=tangent.linear(lambda t: dagger(t)),
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
    jvp=tangent.linear(lambda t, index, c: element(t, index)),
)

_scatter = primitive(
    "scatter",
    _plain_scatter,
    lambda v, index, c: c.copy(),
    vjp=lambda i, flow, v, index, c: element(flow, index),
    reads=((),),
    jvp=tangent.linear(lambda t, index, c: scatter(t, index, c)),
)


def element(a, index):
    # a[index] (plain or a node)
    return _element(a, index=index, c=get_container(a).copy())


def scatter(v, index, c):
    # the zero of the container c with v at index
    return _scatter(v, index=index, c=c)
