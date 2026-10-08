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
from gpt.ad.reverse.util import (
    get_container,
    get_unary_container,
    get_mul_container,
    get_div_container,
    convert_container,
    product,
    add,
    sub,
    div,
    value_of,
    nodify,
    is_node,
    record,
    _structural,
    differentiable,
)
from gpt.ad.reverse import flow as flows
from gpt.ad.reverse.flow import accum, accum_element
from gpt.ad.reverse import foundation
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


def str_traverse(node, indent=0):
    if not callable(node._forward):
        return (" " * indent) + "leaf(" + str(node._container) + ")"
    else:
        pre = " " * indent
        if node._tag is not None:
            tag = node._tag
        else:
            tag = str(node._forward)
        ret = pre + "(" + tag + "):"
        for x in node._children:
            ret = ret + "\n" + str_traverse(x, indent + 1)
        return ret


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
            if _structural:
                # (built during a pass restricted by wrt, e.g. a recorded
                # flow: from the children's structural flags, see _select)
                with_gradient = any([differentiable(c) for c in _children])
            else:
                with_gradient = any([c.with_gradient for c in _children])
        self.with_gradient = with_gradient
        self.infinitesimal_to_cartesian = infinitesimal_to_cartesian
        # the typed gradient (see flow.py); .gradient is its value
        self.flow = None
        self._tag = _tag
        # which values the backward reads (see needed_values): _reads_children
        # is None (all children) or, per child i, the indices of the children
        # whose values the flow into child i reads; _reads_self: whether the
        # backward reads this node's own value.  The default is conservative.
        self._reads_children = None
        self._reads_self = True

    def __str__(self):
        return str_traverse(self)

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

        if x.with_gradient:
            x = convert_container(
                x, z_container, y._container, lambda a, b: a * g.adj(b), "a*adj(b)"
            )

        if y.with_gradient:
            y = convert_container(
                y, x._container, z_container, lambda a, b: g.adj(a) * b, "adj(a)*b"
            )

        return node_op(
            (x, y),
            lambda: product(value_of(x), value_of(y)),
            (
                lambda z: (1, product(z.gradient, g.adj(value_of(y)))),
                lambda z: (1, product(g.adj(value_of(x)), z.gradient)),
            ),
            z_container,
            "*",
            reads=((1,), (0,)),
        )

    def __pow__(x, n):
        x = nodify(x)

        assert g.util.is_num(n)

        z_container = x._container

        # lattice data has no C++ power op, so integer powers are built from
        # repeated multiplication (field * field is supported); scalars and
        # tensors keep the native **.  In a recorded pass value_of(x) is the
        # node and * dispatches on the type, so the product stays a node.
        lattice = x._container.tag[0] == g.lattice

        def _p(v, k):
            if not lattice:
                return v ** k
            if k == 0:
                return 1
            r = v
            for _ in range(k - 1):
                r = r * v
            return r

        # backprop second factor: z = x**n -> dz/dx = n*x**(n-1).  The
        # framework gradient is conjugate-linear (see __mul__, which applies
        # g.adj to the cofactor), so the contribution is adj(n*x**(n-1)) * flow.
        # This holds for both lattice and scalar data; for real values the adj
        # is the identity, so real-data results are unchanged.
        def _bp(v):
            return g.adj(_p(v, n - 1))

        # z = x**n -> dz = n*x**(n-1) dx
        return node_op(
            (x,),
            lambda: _p(value_of(x), n),
            (lambda z: (1, product(z.gradient * n, _bp(value_of(x)))),),
            z_container,
            "**",
            reads=((0,),),
        )

    def __rmul__(x, y):
        return node_base.__mul__(y, x)

    def __truediv__(x, y):
        x, y = nodify(x, y)

        z_container = get_div_container(x._container, y._container)

        # z = x / y -> dz = dx/y - x/y^2 dy.  The cofactor to x is 1/y
        # (pointwise); the cofactor to y is -x/y^2, a lattice while y is a
        # scalar, so its contribution is a contraction (inner_product is
        # conjugate-linear in the cofactor, which is where adj is applied).
        return node_op(
            (x, y),
            lambda: div(value_of(x), value_of(y)),
            (
                lambda z: (1, div(z.gradient, g.adj(value_of(y)))),
                lambda z: (
                    -1,
                    g.inner_product(value_of(x) / value_of(y) ** 2, z.gradient),
                ),
            ),
            z_container,
            "/",
            reads=((1,), (0, 1)),
        )

    def __neg__(self):
        return (-1.0) * self

    def __getitem__(x, item):
        # list node (e.g. the 4 gauge links): element access.  The element's
        # gradient accumulates into its entry of the list's gradient
        if x._container.tag[0] is list:
            def _forward():
                return value_of(x)[item]

            def _backward(z):
                if x.with_gradient:
                    accum_element(x, item, z.gradient)

            z_container = get_unary_container(
                x._container, lambda y: y[item], ("getitem", repr(item))
            )
            z = node_base(_forward, _backward, (x,), _container=z_container)
            z._reads_children = ((),)
            z._reads_self = False
            return z

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

        if not x._container.accumulate_compatible(y._container):
            raise Exception(
                f"Containers incompatible in addition: {x._container} and {y._container}"
            )
        _container = x._container

        return node_op(
            (x, y),
            lambda: add(value_of(x), value_of(y)),
            (lambda z: (1, z.gradient), lambda z: (1, z.gradient)),
            _container,
            "+",
            reads=((), ()),
        )

    def __sub__(x, y):
        x, y = nodify(x, y)

        assert x._container == y._container
        _container = x._container

        return node_op(
            (x, y),
            lambda: sub(value_of(x), value_of(y)),
            (lambda z: (1, z.gradient), lambda z: (-1, z.gradient)),
            _container,
            "-",
            reads=((), ()),
        )

    def __rsub__(x, y):
        return node_base.__sub__(y, x)

    def __radd__(x, y):
        return node_base.__add__(y, x)

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

    def _reverse(
        self, nodes, first_gradient, initial_gradient, retain_values=False, create_graph=False
    ):
        # create_graph: the backward closures are recorded (see util.record):
        # they see the children nodes instead of their values, so the flows,
        # and the leaf gradients, are nodes of the same graph (nothing is
        # evaluated); a contraction of a leaf gradient is then an ordinary
        # node whose reverse pass gives the next derivative
        with record(create_graph):
            self._reverse_pass(nodes, first_gradient, initial_gradient, retain_values)

    def _reverse_pass(self, nodes, first_gradient, initial_gradient, retain_values):
        if initial_gradient is None:
            if self._container.is_field():
                raise Exception(
                    "Expression evaluates to a field.  Gradient calculation is not unique."
                )
            initial_gradient = 1.0
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
        # create_graph=True: record the reverse pass (see _reverse); the leaf
        # gradients are then lazy nodes.  The recording reads no values, so
        # with with_value=False no forward runs at all
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
                self._reverse(
                    nodes,
                    first_gradient=forward_free,
                    initial_gradient=initial_gradient,
                    retain_values=retain_values,
                    create_graph=create_graph,
                )
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
        self._container.set_otype(v)

    def get_real(self):
        # Re x (componentwise; the flow into x is the real part of the flow)
        return g.ad.reverse.transform.real(self)

    grid = property(get_grid)
    otype = property(get_otype, set_otype)
    real = property(get_real)

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


def node_op(children, forward, backards, container, tag=None, reads=None):
    # reads: None (conservative) or, per child i, the indices of the children
    # whose values the backward closure of child i reads; a node_op's backward
    # never reads its own value
    # build a node from a forward closure and per-child backward closures.
    # backards[i](z) returns (sign, term) or None (no gradient for that
    # child); the with_gradient check and gradient accumulation are handled
    # here.  Backward closures receive z as an argument and must not capture
    # it, otherwise there is a reference loop.
    def _backward(z):
        for c, f in zip(children, backards):
            if f is not None and c.with_gradient:
                sign, term = f(z)
                accum(c, term, sign)

    z = node_base(forward, _backward, children, _container=container, _tag=tag)
    if reads is not None:
        z._reads_children = tuple(tuple(r) for r in reads)
        z._reads_self = False
    return z


def zero(container):
    # a lazy zero of the container (a constant; built only if evaluated)
    return node_base(container.zero, _container=container, with_gradient=False)


def node(x, with_gradient=True, infinitesimal_to_cartesian=True):
    return node_base(
        x, with_gradient=with_gradient, infinitesimal_to_cartesian=infinitesimal_to_cartesian
    )
