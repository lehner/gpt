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
import gpt as g
import numpy as np
from gpt.ad.reverse.util import (
    get_container,
    get_mul_container,
    get_div_container,
    infer_container,
    container_reductions,
    value_of,
    nodify,
    is_node,
    record,
    _structural,
    gradient_flag,
)
from gpt.ad.reverse import flow as flows
from gpt.ad.reverse.flow import accum
from gpt.ad.reverse.primitive import primitive, forget
from gpt.ad.reverse import tangent
from gpt.ad.reverse import foundation
from gpt.ad.reverse.node_str import node_str
from gpt.core.foundation import base

def traverse(nodes, root):
    # appends the nodes of the graph of root to nodes in topological order
    # (children first, depth-first in child order; iterative, since recorded
    # graphs are deep) and returns, per node, the nodes whose last use it is
    # (when their values can be released in a forward pass)
    visited = {root}
    stack = [(root, iter(root._children))]
    while stack:
        n, children = stack[-1]
        for c in children:
            if c not in visited:
                visited.add(c)
                stack.append((c, iter(c._children)))
                break
        else:
            stack.pop()
            nodes.append(n)

    last_need = {}
    for n in nodes:
        for x in n._children:
            last_need[x] = n

    forward_free = dict([(x, []) for x in nodes])
    for x in last_need:
        forward_free[last_need[x]].append(x)
    return forward_free


def _left_operand(y):
    # a plain operand on the left of a node; a numpy scalar (or 0-d array)
    # is the Python number (numpy defers to the node, see __array_ufunc__)
    if isinstance(y, np.generic) or (isinstance(y, np.ndarray) and y.ndim == 0):
        return y.item()
    return y


def _check_compatible(x, y, operation):
    # the operands of a sum or difference (the result has the container of x)
    if not x._container.accumulate_compatible(y._container):
        raise Exception(
            f"Containers incompatible in {operation}: {x._container} and {y._container}"
        )


def _is_zero(x):
    return g.util.is_num(x) and x == 0


class node_differentiable_functional(g.group.differentiable_functional):
    def __init__(self, node, arguments):
        self.node = node
        self.arguments = arguments

    def __call__(self, fields):
        assert len(fields) == len(self.arguments)
        for i in range(len(fields)):
            self.arguments[i].value = fields[i]
        return self.node(with_gradients=False).real

    def gradient(self, fields, dfields):
        indices = [g.util.index_by_identity(fields, df) for df in dfields]
        for i in range(len(fields)):
            self.arguments[i].value = fields[i]
        return self.node.backward(wrt=[self.arguments[i] for i in indices])


def _check_value(v):
    # the value of a node is a plain value (lattice, tensor, number, array, a
    # list of these) or a forward-AD series, never a node: higher reverse
    # orders are recorded passes over one graph (create_graph), not nested
    # nodes
    if isinstance(v, node_base):
        raise ValueError(
            "the value of a node is a plain value or a forward-AD series, not a node "
            "(higher derivatives: y.backward(create_graph=True))"
        )


class _value_slot:
    # node.value: read as a plain attribute (no __get__), checked on writes;
    # the one place where this is enforced for every writer (forward,
    # functionals and other leaf swaps)
    def __set__(self, n, v):
        _check_value(v)
        n.__dict__["value"] = v


class node_base(base):
    value = _value_slot()
    foundation = foundation
    # (numpy defers to the reflected operators: array * node is a node, see
    # _left_operand)
    __array_ufunc__ = None

    # TODO: deprecate infinitesimal_to_cartesian and make it default
    def __init__(
        self,
        _forward,
        _backward=lambda z: None,
        _children=(),
        with_gradient=True,
        infinitesimal_to_cartesian=True,
        _container=None,
        _tag=None,
        _name=None,
    ):
        if not callable(_forward) or isinstance(_forward, node_base):
            # a leaf with the value _forward
            _check_value(_forward)
            self._forward = None
            self.__dict__["value"] = _forward
            _container = get_container(_forward)
        else:
            self._forward = _forward
            self.__dict__["value"] = None
            assert _container is not None
        self._container = _container
        self._backward = _backward
        self._children = _children
        if len(_children) > 0:
            with_gradient = gradient_flag(_children)
        self.with_gradient = with_gradient
        self.infinitesimal_to_cartesian = infinitesimal_to_cartesian
        # the typed gradient (see flow.py); .gradient is its value
        self.flow = None
        self._tag = _tag
        # a name for printing (see node_str)
        self._name = _name
        # which values the backward reads (see needed_values): _reads_children
        # is None (all children) or, per child i, the indices of the children
        # whose values the flow into child i reads; _reads_self: whether the
        # backward reads this node's own value.  The default is conservative.
        self._reads_children = None
        self._reads_self = True
        # the tangent rule (see g.ad.reverse.jacobian): None, or
        # _jvp(z, children, tangents) -> the k tangents of z, given per child
        # None (constant) or its k tangents; z and children are the nodes that
        # stand for the values (the node's own, or a replay's copies)
        self._jvp = None

    def __str__(self):
        return node_str(self)

    @property
    def gradient(self):
        # the value of the flow (None: no flow); a scaled identity is built
        # into a field on reading
        if isinstance(self.flow, flows.scaled_identity):
            self.flow = flows.built(self.flow, self._container)
        return flows.value(self.flow)

    @gradient.setter
    def gradient(self, v):
        self.flow = flows.wrap(v, self._container)

    def own_gradient(self):
        # make the plain gradient (list elements) exclusively owned: copies of
        # adopted fields, e.g., before it is updated in place or handed out
        self.flow = flows.owned(self.flow)

    def materialize_gradient(self):
        # a missing flow (or list element) is a zero not built yet
        if self.flow is None:
            self.zero_gradient()
        elif isinstance(self.flow, flows.flow_list) and None in self.flow.elements:
            elem = self._container.tag[1]
            self.flow = flows.flow_list(
                [flows.dense(elem.zero(), True) if e is None else e for e in self.flow.elements]
            )
        else:
            self.gradient

    def zero_gradient(self):
        if self._container.tag[0] is list:
            # a list leaf's gradient is a plain list, one entry per element
            # (in a recorded pass a list of node graphs, never a node
            # wrapping a list)
            elem = self._container.tag[1]
            self.gradient = [elem.zero() for _ in range(self._container.tag[2])]
            return
        if isinstance(self.value, g.ad.forward.series):
            zero = self._container.zero()
            self.gradient = 0.0 * self.value
            for t in self.gradient.terms:
                self.gradient.terms[t] = zero
            return
        self.gradient = self._container.zero()

    def __mul__(x, y):
        x, y = nodify(x, y)

        z_container = get_mul_container(x._container, y._container)

        # (a factor whose flow has another type than the factor, e.g. a
        # number times a field, receives the flow reduced to its type)
        if x.with_gradient:
            x = _converted(x, ("a*adj(b)", (z_container, y._container), lambda a, b: a * g.adj(b)))

        if y.with_gradient:
            y = _converted(y, ("adj(a)*b", (x._container, z_container), lambda a, b: g.adj(a) * b))

        return _mul.node(x, y, c=z_container)

    def __pow__(x, n):
        x = nodify(x)

        assert g.util.is_num(n)

        # lattice data has no C++ power op, so integer powers are built from
        # repeated multiplication (field * field is supported); scalars and
        # tensors keep the native **.  Matrix fields do not commute, see
        # _pow_vjp
        lattice = x._container.tag[0] == g.lattice
        otype = x._container.get_otype() if lattice else None
        matrix = otype is not None and len(otype.shape) == 2 and otype.shape[0] > 1
        return _pow.node(x, n=n, lattice=lattice, matrix=matrix)

    def __rmul__(x, y):
        return node_base.__mul__(_left_operand(y), x)

    def __truediv__(x, y):
        return _div.node(*nodify(x, y))

    def __neg__(self):
        return (-1.0) * self

    def __getitem__(x, item):
        # list node (e.g. the 4 gauge links): element access.  The element's
        # gradient accumulates into its entry of the list's gradient
        if x._container.tag[0] is list:
            n = len(x)
            return _list_element.node(x, index=range(n)[item], n=n)

        # element access (arrays, tensors, ...; see linear.element), to any
        # order (its vjp is the scatter, whose vjp is again element)
        return g.ad.reverse.linear.element(x, item)

    def __len__(self):
        # a list node (e.g. the 4 gauge links) reports its length so it can be
        # consumed like the Python-list-of-nodes convention (len(U), U[i])
        if self._container.tag[0] is list:
            return self._container.tag[2]
        raise TypeError(f"object of type '{type(self).__name__}' has no len()")

    def __add__(x, y):
        # an exact numeric zero is the neutral element (so that python's sum,
        # which starts from 0, works on nodes)
        if _is_zero(y):
            return x
        if _is_zero(x):
            return y
        x, y = nodify(x, y)
        _check_compatible(x, y, "addition")
        return _add.node(x, y)

    def __sub__(x, y):
        x, y = nodify(x, y)
        _check_compatible(x, y, "subtraction")
        return _sub.node(x, y)

    def __rsub__(x, y):
        return node_base.__sub__(_left_operand(y), x)

    def __radd__(x, y):
        return node_base.__add__(_left_operand(y), x)

    def forward(self, nodes, free=None, needed=None):
        for n in nodes:
            if n._forward is not None:
                if needed is not None and n not in needed:
                    # no backward reads this value (see needed_values); a
                    # read that was not declared evaluates it lazily
                    continue
                if n.value is None or free is not None:
                    # in a backward pass (free is None) a value that survived
                    # a previous pass (retain_values) is kept instead of
                    # re-computed: the caller retains values only while the
                    # leaves are unchanged.  In a forward-only pass the same
                    # graph may be re-evaluated with modified leaf values, so
                    # values are re-computed
                    n.value = n._forward()
                if free is not None:
                    for m in free[n]:
                        if m._forward is not None:
                            m.value = None
                if isinstance(n.value, g.expr):
                    if not n.value.is_adj():
                        n.value = g(n.value)

    def _reverse(self, nodes, first_gradient, initial_gradient, retain_values):
        if initial_gradient is None:
            if self._container.is_field():
                raise Exception(
                    "Expression evaluates to a field.  Gradient calculation is not unique."
                )
            initial_gradient = 1.0
        if g.util.is_num(initial_gradient) and self._container.tag[0] is g.tensor:
            # a number seeds a tensor of a single component (e.g. a U(1)
            # element, which is not cast to a number, see container)
            otype = self._container.get_otype()
            if otype.shape != (1,):
                raise Exception(
                    "Expression evaluates to a tensor.  Gradient calculation is not unique."
                )
            initial_gradient = g.tensor(np.array([initial_gradient], dtype=np.complex128), otype)
        # a gradient of None is a zero that is not built (see accumulate)
        # (never adopt the caller's initial gradient: it could come back as a
        # leaf gradient or be updated in place)
        self.flow = None
        accum(self, initial_gradient, adopt=False)
        for n in reversed(nodes):
            for m in first_gradient[n]:
                if m is not self:
                    m.flow = None
            if n.flow is not None:
                # (a zero flow contributes nothing to the children)
                n._backward(n)
            if n._forward is not None:
                n.flow = None
                if n is not self and not retain_values:
                    n.value = None
            elif n.with_gradient:
                # (constant leaves keep gradient None: nothing reads it)
                n.materialize_gradient()
                if n.infinitesimal_to_cartesian:
                    # (value_of: while recording, the leaf node itself, so
                    # that the conversion is differentiated)
                    n.flow = flows.replaced(
                        n.flow, g.infinitesimal_to_cartesian(value_of(n), n.gradient)
                    )
                # (a gradient handed out never aliases another gradient)
                n.own_gradient()

    # TODO: allow for lists of initial_gradients (could save forward runs at sake of more memory)
    def __call__(
        self,
        with_gradients=True,
        initial_gradient=None,
        retain_values=False,
        with_value=True,
        create_graph=False,
        wrt=None,
    ):
        # wrt: a list of leaves; the pass computes their gradients only (the
        # other leaves are constants for this pass and keep their gradients)
        # y() runs the forward and the reverse pass and returns the value of y;
        # y.backward() is the same without the value (see there).
        # create_graph=True: record the reverse pass: the backward closures
        # see the children nodes instead of their values (util.record), so
        # the flows are nodes of the same graph and a contraction of a leaf
        # gradient is an ordinary node whose reverse pass gives the next
        # derivative.  The recording reads no values, so with
        # with_value=False no forward runs at all
        # with_value=False (with gradients, without retain_values): the value
        # of the root is not needed, so only the values some backward reads
        # are computed (see needed_values); the return value is then None
        # retain_values keeps the forward values of the graph (the root's
        # included): with gradients, the backward does not free them, so
        # repeated reverse passes (e.g. one per seed direction) over
        # unchanged leaves share one forward; without, the forward keeps all
        # intermediate values, so a following reverse pass reuses exactly
        # these values.  Without retain_values no value survives the pass
        nodes = []
        forward_free = traverse(nodes, self)
        saved = None
        if with_gradients and wrt is not None:
            saved, added = _select(nodes, wrt)
            for x in wrt:
                # (a selected leaf the root does not depend on: no gradient)
                x.flow = None
        try:
            free = forward_free if not (with_gradients or retain_values) else None
            needed = None
            if with_gradients and not retain_values and not with_value:
                needed = set() if create_graph else needed_values(nodes)
            self.forward(nodes, free=free, needed=needed)
            if with_gradients:
                with record(create_graph):
                    self._reverse(nodes, forward_free, initial_gradient, retain_values)
        finally:
            if saved is not None:
                for n, w in saved:
                    n.with_gradient = w
                for n in added:
                    del _structural[n]
        value = self.value if needed is None else None
        if not retain_values:
            # values survive a pass only with retain_values (else a later
            # pass with modified leaf values would reuse a stale root value)
            self.value = None
        return value

    def backward(
        self,
        initial_gradient=None,
        retain_values=False,
        with_value=False,
        create_graph=False,
        wrt=None,
    ):
        # the gradients without the value of y: y(with_value=False), so the
        # forward computes only the values a backward reads (none at all when
        # recording, create_graph=True).  Use it wherever the value of a
        # reverse pass is not read.  With wrt, returns the gradients of the
        # leaves wrt (as .gradient: a list for a list leaf; None for a leaf
        # the root does not depend on)
        self(
            initial_gradient=initial_gradient,
            retain_values=retain_values,
            with_value=with_value,
            create_graph=create_graph,
            wrt=wrt,
        )
        if wrt is not None:
            return [x.gradient for x in wrt]

    def functional(self, *arguments):
        return node_differentiable_functional(self, arguments)

    def get_grid(self):
        return self._container.get_grid()

    def get_otype(self):
        return self._container.get_otype()

    def set_otype(self, v):
        # (a retyped node no longer stands for the op that built it)
        forget(self)
        self._container.set_otype(v)

    def get_real(self):
        # Re x (componentwise; the flow into x is the real part of the flow)
        return g.ad.reverse.transform.real(self)

    grid = property(get_grid)
    otype = property(get_otype, set_otype)
    real = property(get_real)

    def _become(self, z):
        # this node becomes the computed node z (e.g. the output node a
        # stencil installs its node into): everything that describes the
        # computation is z's; the node keeps its identity, its container, its
        # name and its leaf conversion flag, its old value and flow are
        # discarded
        for name, v in z.__dict__.items():
            if name not in ("value", "flow", "_container", "infinitesimal_to_cartesian", "_name"):
                self.__dict__[name] = v
        self.__dict__["value"] = None
        self.flow = None

    def new(self):
        # a fresh zero of the same type as self (lattice, tensor, number,
        # list, ...), lazy: the output node of a stencil, into which the
        # stencil installs its computed node, so the zero is never built
        return zero(self._container)


def _select(nodes, wrt):
    # restrict a pass to the leaves wrt: the with_gradient flags of the
    # graph's nodes for this pass (a leaf: whether it is in wrt; a computed
    # node: whether it has a selected child), returned with the structural
    # flags for restoring and the nodes it entered into _structural.  wrt
    # only restricts: a leaf constructed as a constant (with_gradient=False)
    # cannot be selected
    for x in wrt:
        if not isinstance(x, node_base) or x._forward is not None:
            raise TypeError("wrt: the leaves of the graph (rad.node(...)) to differentiate")
        if not x.with_gradient:
            raise ValueError(
                "wrt: a leaf constructed with with_gradient=False is a constant; "
                "construct it with with_gradient=True to differentiate it"
            )
    wrt = set(wrt)
    saved = [(n, n.with_gradient) for n in nodes]
    # (a pass inside another restricted pass, e.g. in a backward closure,
    # keeps the outer pass's structural flags)
    added = [n for n, w in saved if n not in _structural]
    for n, w in saved:
        if n not in _structural:
            _structural[n] = w
    for n in nodes:
        # (children first: their flags are already the pass's)
        if n._forward is None:
            n.with_gradient = n in wrt
        elif n.with_gradient:
            n.with_gradient = any(c.with_gradient for c in n._children)
    return saved, added


def needed_values(nodes):
    # the computed nodes whose values a reverse pass reads when the value of
    # the root (nodes[-1]) is not needed: the values the backward closures
    # read (as declared by _reads_children / _reads_self), and everything
    # their forward closures need.  nodes is in topological order (children
    # first), so every consumer of a node is visited before the node.
    needed = set()
    for n in reversed(nodes):
        if n._forward is None:
            continue
        if n.with_gradient:
            if n._reads_self:
                needed.add(n)
            if n._reads_children is None:
                needed.update(n._children)
            else:
                for c, reads in zip(n._children, n._reads_children):
                    if c.with_gradient:
                        needed.update(n._children[j] for j in reads)
        if n in needed:
            needed.update(n._children)
    return needed


# The arithmetic of nodes, as primitives (see primitive.py).  The gradient
# is conjugate-linear: the flow into a factor is the flow times the adjoint
# of its cofactor.


# x * y with the container c of the product
_mul = primitive(
    "*",
    lambda x, y, c: x * y,
    lambda x, y, c: c,
    vjp=lambda i, flow, x, y, c: flow * g.adj(y) if i == 0 else g.adj(x) * flow,
    reads=((1,), (0,)),
    jvp=tangent.bilinear(lambda x, y, c: x * y),
)


def _sum_jvp(sign):
    # z = x + sign y: dz = dx + sign dy (a constant contributes nothing)
    def rule(z, children, tangents):
        (tx, ty) = tangents
        return [
            tangent.total(
                [
                    None if tx is None else tx[j],
                    None if ty is None else (ty[j] if sign == 1.0 else sign * ty[j]),
                ]
            )
            for j in range(tangent.count(tangents))
        ]

    return rule


_add = primitive(
    "+",
    lambda x, y: x + y,
    lambda x, y: x,
    vjp=lambda i, flow, x, y: flow,
    reads=((), ()),
    jvp=_sum_jvp(1.0),
)

_sub = primitive(
    "-",
    lambda x, y: x - y,
    lambda x, y: x,
    vjp=lambda i, flow, x, y: flow if i == 0 else flows.negative(flow),
    reads=((), ()),
    jvp=_sum_jvp(-1.0),
)


def _div_vjp(i, flow, x, y):
    # z = x / y -> dz = dx/y - x/y^2 dy.  The cofactor to x is 1/y
    # (pointwise); the cofactor to y is -x/y^2, a lattice while y is a
    # scalar, so its contribution is a contraction (inner_product is
    # conjugate-linear in the cofactor, which is where adj is applied)
    if i == 0:
        return flow / g.adj(y)
    return flows.negative(g.inner_product(x / y**2, flow))


_div = primitive(
    "/",
    lambda x, y: x / y,
    get_div_container,
    vjp=_div_vjp,
    reads=((1,), (0, 1)),
)


def _power(v, k, lattice):
    # v**k (repeated multiplication for lattices; in a recorded pass v is a
    # node and * dispatches on the type, so the product stays a node)
    if not lattice:
        return v**k
    if k == 0:
        return 1
    r = v
    for _ in range(k - 1):
        r = r * v
    return r


def _between(left, middle, right, mul):
    # left middle right with the factors of power 0 left out
    r = middle
    if left is not None:
        r = mul(left, r)
    if right is not None:
        r = mul(r, right)
    return r


def _pow_vjp(i, flow, x, n, lattice, matrix):
    # z = x**n -> dz/dx = n*x**(n-1), so the flow is adj(n*x**(n-1)) * flow
    # (for real values the adj is the identity).  For matrices the flow is
    # sum_i adj(x^i) flow adj(x^(n-1-i))
    if not matrix:
        return flow * n * g.adj(_power(x, n - 1, lattice))
    return tangent.total(
        [
            _between(
                g.adj(_power(x, j, lattice)) if j > 0 else None,
                flow,
                g.adj(_power(x, n - 1 - j, lattice)) if j < n - 1 else None,
                lambda a, b: a * b,
            )
            for j in range(n)
        ]
    )


def _pow_jvp(z, children, tangents, n, lattice, matrix):
    # z = x**n -> dz = n*x**(n-1) dx (matrices: sum_i x^i dx x^(n-1-i))
    (x,), (t,) = children, tangents
    if n == 0:
        return tangent.constant(z, children, tangents)
    if not matrix:
        return [n * _power(x, n - 1, lattice) * dx for dx in t]
    return [
        tangent.total(
            [
                _between(
                    _power(x, j, lattice) if j > 0 else None,
                    dx,
                    _power(x, n - 1 - j, lattice) if j < n - 1 else None,
                    lambda a, b: a * b,
                )
                for j in range(n)
            ]
        )
        for dx in t
    ]


_pow = primitive(
    "**",
    lambda x, n, lattice, matrix: _power(x, n, lattice),
    lambda x, n, lattice, matrix: x,
    vjp=_pow_vjp,
    reads=((0,),),
    jvp=_pow_jvp,
)


# element index of a list node of n elements: the flow into the list is the
# flow at index (no flow into the other elements)
_list_element = primitive(
    "list_element",
    lambda x, index, n: x[index],
    lambda x, index, n: x.tag[1].copy(),
    vjp=lambda i, flow, x, index, n: [flow if j == index else None for j in range(n)],
    reads=((),),
    jvp=tangent.linear(lambda t, index, n: t[index]),
)


def _reduced(flow, reductions):
    # the flow reduced by the functions reductions (names of g, in order)
    for name in reductions:
        flow = getattr(g, name)(flow)
    if any(name != "sum" for name in reductions) and not is_node(flow):
        flow = g(flow)
    return flow


# v in the container c (the same value; the flow is reduced to v's type)
_convert = primitive(
    "convert",
    lambda v, c, reductions: v,
    lambda v, c, reductions: c,
    vjp=lambda i, flow, v, c, reductions: _reduced(flow, reductions),
    reads=((),),
    # (the conversion is linear: the tangents are converted alike)
    jvp=tangent.linear(lambda t, c, reductions: _convert(t, c=c, reductions=reductions)),
)


def _converted(v, inferred):
    # v converted to the container infer_container(*inferred) (v itself if it
    # has that type already)
    c = infer_container(*inferred)
    reductions = container_reductions(v._container, c)
    if not reductions:
        return v
    return _convert.node(v, c=c, reductions=reductions)


def zero(container):
    # a lazy zero of the container (a constant; built only if evaluated)
    return node_base(container.zero, _container=container, with_gradient=False)


def node(x, with_gradient=True, infinitesimal_to_cartesian=True, name=None):
    # name: how the leaf is printed (str(node), see node_str)
    return node_base(
        x,
        with_gradient=with_gradient,
        infinitesimal_to_cartesian=infinitesimal_to_cartesian,
        _name=name,
    )
