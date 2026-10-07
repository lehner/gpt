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
    zero_of,
    nodify,
)
from gpt.ad.reverse import flow as flows
from gpt.ad.reverse.flow import accum, accum_element
from gpt.ad.reverse import foundation
from gpt.core.foundation import base

verbose_memory = g.default.is_verbose("ad_memory")


def traverse(nodes, n, visited=None):
    # forward(children) = value
    # last usage
    root = visited is None
    if root:
        visited = set([])
    if n not in visited:
        visited.add(n)
        for c in n._children:
            traverse(nodes, c, visited)
        nodes.append(n)

    if root:
        last_need = {}
        for n in nodes:
            for x in n._children:
                last_need[x] = n

        forward_free = dict([(x, []) for x in nodes])
        for x in last_need:
            forward_free[last_need[x]].append(x)

        # forward free contains information for when we can
        # release the node.value for forward propagation
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
        for a in self.arguments:
            a.with_gradient = False
        indices = [g.util.index_by_identity(fields, df) for df in dfields]
        for i in indices:
            self.arguments[i].gradient = None
            self.arguments[i].with_gradient = True
        for i in range(len(fields)):
            self.arguments[i].value = fields[i]
        self.node(with_value=False)
        return [self.arguments[i].gradient for i in indices]


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


class _value_slot:
    # node.value: read as a plain attribute (no __get__), checked on writes.
    # A node's depth is fixed at construction, so a value must have the
    # depth below it (None: no value); the one place where this is enforced
    # for every writer (forward, functionals and other leaf swaps)
    def __set__(self, n, v):
        if v is not None and _depth(v) != n.depth - 1:
            raise ValueError(
                f"a value of depth {_depth(v)} assigned to a node of depth {n.depth} "
                f"(the value of a node of depth d has depth d - 1; node depths are "
                f"fixed at construction)"
            )
        n.__dict__["value"] = v


def _depth(x):
    return x.depth if isinstance(x, node_base) else 0


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
        # depth: the number of nested node levels, fixed at construction (see
        # _value_slot): a leaf is one level above its value; a computed node
        # has the depth of its deepest child (a node op promotes shallower
        # operands, e.g. constants, to the depth of the others)
        if not callable(_forward) or isinstance(_forward, node_base):
            self._forward = None
            self.depth = 1 + _depth(_forward)
            self.__dict__["value"] = _forward
            _container = get_container(_forward)
        else:
            self._forward = _forward
            self.depth = max([c.depth for c in _children], default=1)
            self.__dict__["value"] = None
            assert _container is not None
        self._container = _container
        self._backward = _backward
        self._children = _children
        if len(_children) > 0:
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
            self.flow = flows.built(
                self.flow, self._container, lambda: self.depth - 1
            )
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
            depth = self.depth - 1
            elem = self._container.tag[1]
            self.flow = flows.flow_list(
                [flows.dense(zero_of(elem, depth), True) if e is None else e for e in self.flow.elements]
            )
        else:
            self.gradient

    def zero_gradient(self):
        depth = self.depth - 1
        if self._container.tag[0] is list:
            # a list leaf's gradient is a plain list, one entry per element,
            # each element independently wrapped to the nesting depth -- so at
            # a nested depth the gradient is a list of node graphs (one per
            # element), never a node wrapping a list
            elem = self._container.tag[1]
            self.gradient = [zero_of(elem, depth) for _ in range(self._container.tag[2])]
            return
        if isinstance(self.value, g.ad.forward.series):
            # (a series value is plain, depth 0)
            zero = self._container.zero()
            self.gradient = 0.0 * self.value
            for t in self.gradient.terms:
                self.gradient.terms[t] = zero
            return
        self.gradient = zero_of(self._container, depth)

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
        # tensors keep the native **.  In a nested pass value_of(x) is a node
        # and * dispatches on the type, so the product stays in the node world.
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

        # element access (arrays, tensors, ...; see linear.element), at any
        # nesting depth (its vjp is the scatter, whose vjp is again element)
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
        max_fields_allocated = 0
        fields_allocated = 0
        for n in nodes:
            if n._forward is not None:
                if needed is not None and n not in needed:
                    # no backward reads this value (see needed_values); a
                    # read that was not declared evaluates it lazily
                    continue
                if n.value is None or free is not None:
                    # in a backward pass (free is None) a node's value is a
                    # deterministic function of its children's values, which
                    # are immutable; nested passes re-enter the previous
                    # pass's graph, so a value that survived the backward
                    # (e.g. the root, which is never freed) is kept instead of
                    # re-computed.  In a forward-only pass the same graph may
                    # be re-evaluated with modified leaf values, so values are
                    # re-computed as before
                    n.value = n._forward()
                    fields_allocated += 1
                    max_fields_allocated = max(max_fields_allocated, fields_allocated)
                if free is not None:
                    free_n = free[n]
                    for m in free_n:
                        if m._forward is not None:
                            m.value = None
                            fields_allocated -= 1
                if isinstance(n.value, g.expr):
                    if not n.value.is_adj():
                        n.value = g(n.value)

        if verbose_memory:
            g.message(
                f"Forward propagation through graph with {len(nodes)} nodes with maximum allocated fields: {max_fields_allocated}"
            )

    def backward(self, nodes, first_gradient, initial_gradient, retain_values=False):
        fields_allocated = len(nodes)  # .values
        max_fields_allocated = fields_allocated
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
            first_gradient_n = first_gradient[n]
            for m in first_gradient_n:
                if m is not self:
                    m.flow = None
                    fields_allocated += 1
                    max_fields_allocated = max(max_fields_allocated, fields_allocated)
            if n.flow is not None:
                # (a zero flow contributes nothing to the children)
                n._backward(n)
            if n._forward is not None:
                n.flow = None
                fields_allocated -= 1
                if n is not self and not retain_values:
                    n.value = None
                    fields_allocated -= 1
            elif n.with_gradient:
                # (constant leaves keep gradient None: nothing reads it)
                n.materialize_gradient()
                if n.infinitesimal_to_cartesian:
                    n.flow = flows.replaced(
                        n.flow, g.infinitesimal_to_cartesian(n.value, n.gradient)
                    )
                # (a gradient handed out never aliases another gradient)
                n.own_gradient()

        if verbose_memory:
            g.message(
                f"Backward propagation through graph with {len(nodes)} nodes with maximum allocated fields: {max_fields_allocated}"
            )

    # TODO: allow for lists of initial_gradients (could save forward runs at sake of more memory)
    def __call__(
        self, with_gradients=True, initial_gradient=None, retain_values=False, with_value=True
    ):
        # with_value=False (with gradients, without retain_values): the value
        # of the root is not needed, so only the values some backward reads
        # are computed (see needed_values); the return value is then None
        # retain_values keeps the forward values of the graph: with gradients,
        # the backward does not free them, so repeated reverse passes (e.g.
        # one per seed direction) over unchanged leaves share one forward;
        # without, the forward keeps all intermediate values, so a following
        # reverse pass reuses exactly these nodes (in a nested graph, the
        # returned value is then part of the next pass's derivative graph)
        nodes = []
        forward_free = traverse(nodes, self)
        free = forward_free if not (with_gradients or retain_values) else None
        needed = None
        if with_gradients and not retain_values and not with_value:
            needed = needed_values(nodes)
        self.forward(nodes, free=free, needed=needed)
        if with_gradients:
            self.backward(
                nodes,
                first_gradient=forward_free,
                initial_gradient=initial_gradient,
                retain_values=retain_values,
            )
        return self.value if needed is None else None

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
        # a fresh node at the SAME DEPTH as self, zero-initialized, for any
        # node value type (lattice, tensor, number, list, ...).  The container
        # is the innermost one, so it builds the plain zeroed value and one
        # node wraps it per nesting level.  This is what lets
        # parallel_transport_matrix.__call__ allocate a target whose depth
        # matches the (possibly 2nd/3rd-derivative) input.  The producer (e.g.
        # a stencil) overwrites the contents, so zero-init is fine.
        r = self._container.zero()
        for _ in range(self.depth):
            r = node(r)
        return r


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


def node(x, with_gradient=True, infinitesimal_to_cartesian=True):
    return node_base(
        x, with_gradient=with_gradient, infinitesimal_to_cartesian=infinitesimal_to_cartesian
    )
