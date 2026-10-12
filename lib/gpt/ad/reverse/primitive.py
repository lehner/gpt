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
# Primitives: operations closed under differentiation.
#
#   op = primitive(name, plain, container, vjp=...)      (or joint_vjp=...)
#   y = op(*args, **static)
#
# A primitive is given by its plain implementation and its vector-Jacobian
# product, written in terms of primitives (itself or others).  Called on
# plain arguments it runs plain(*args, **static).  Called with a node among
# its arguments it returns a node whose
#
#   forward  is the implementation on the values of the children
#   backward is the vjp at the values of the children
#
# Since the vjp is again made of primitives, it is evaluated in the same way:
# in a plain pass it runs plain implementations; in a recorded pass
# (create_graph) it sees the children nodes instead of their values and
# builds nodes of the same graph.  So every primitive is differentiable to
# any order without separate plain and recording code (the self-similar
# derivative towers of exp and of the stencils).
#
#   plain(*args, **static)                -> the value on plain arguments
#   container(*containers, **static)      -> the container of the value
#   vjp(i, flow, *values, **static)       -> the flow into child i (None: no
#                                            flow; flow.negative(r): -r), or
#   joint_vjp(z, needed, *values, **static) -> {i: flow} for all children i
#                                            in needed at once (z: the node,
#                                            whose gradient is the flow)
#   lift      how an argument becomes a child node (one callable for all
#             arguments, or one per argument; default: constant)
#   reads     per child i, the indices of the children whose values the
#             flow into child i reads (None: all); the vjp receives the
#             values of these children only (the others are None), so an
#             undeclared read fails instead of evaluating a value lazily.
#             A callable reads(n) for a variable number of children.
#   fwd(*values, **static) -> (value, residual): the plain evaluation of a
#             node's value, with a residual that is handed to the vjp of that
#             node (keyword residual; once: None in later passes)
#   order     the highest supported derivative order (None: any); with
#             order=1 a recorded pass raises NotImplementedError
#   jvp(z, children, tangents, **static) -> the k tangents of z: the
#             optional tangent rule used by g.ad.reverse.jacobian, given per
#             child None (constant) or its k tangents, written in primitives
#             (see node_base._jvp)
#   involution  op(op(x)) = x: op of a node built by op is its child
#   fresh     every call builds a new node (not shared, see below), e.g. a
#             node that marks a boundary by its identity
#
# The static keyword arguments are not differentiated; they are passed on
# to plain, container, fwd, the vjp, and to the nodes the vjp builds.
#
# Shared nodes: op(*children, **static) of the same op on the same children
# (by identity) with the same statics is the node built before, as long as it
# is alive, holds no value (a value retained by a pass on another graph may
# be stale for this one) and has the same gradient flag.  So a computation
# built twice (by the user, or by the vjps of a recorded pass, which build
# the same cofactors once per consumer) is evaluated once.  A plain argument
# becomes a new constant node, so it is not shared.
#
import weakref
import gpt as g
from gpt.ad.reverse.util import constant, is_node, value_of, recording, gradient_flag, container
from gpt.ad.reverse.flow import accum

# switches (for comparisons and debugging) and counters of the shared and
# simplified nodes
share = True
simplify = True
stats = {"shared": 0, "simplified": 0}

# (op, ids of the children, statics) -> node; the node holds its children and
# statics, so the ids in a live entry are unique
_nodes = weakref.WeakValueDictionary()


_plain_types = (int, float, complex, str, bool, type(None))


def _static_key(static):
    # (in the order given: another order only means that a node is not
    # shared)
    if not static:
        return ()
    key = []
    for k, v in static.items():
        t = type(v)
        if t in _plain_types:
            # (the type too: 2 and 2.0 are equal but not interchangeable)
            key.append((k, t, v))
        elif t is container:
            # (containers compare by str)
            key.append((k, str(v)))
        else:
            try:
                hash(v)
                key.append((k, t, v))
            except TypeError:
                key.append((k, id(v)))
    return tuple(key)


def forget(z):
    # z is no longer shared (e.g. retyped)
    key = z.__dict__.get("_key")
    if key is not None and _nodes.get(key) is z:
        del _nodes[key]


def has_node(x):
    # a node, or a (nested) list containing one
    if isinstance(x, (list, tuple)):
        return any(has_node(y) for y in x)
    return is_node(x)


def _tuples(reads):
    return tuple(tuple(r) for r in reads)


class primitive:
    def __init__(
        self,
        name,
        plain,
        container,
        vjp=None,
        joint_vjp=None,
        lift=constant,
        reads=None,
        fwd=None,
        order=None,
        jvp=None,
        involution=False,
        fresh=False,
    ):
        assert (vjp is None) != (joint_vjp is None), "a primitive has a vjp or a joint_vjp"
        self.name = name
        self.plain = plain
        self.container = container
        self.vjp = vjp
        self.joint_vjp = joint_vjp
        self.lift = lift
        self.reads = reads if reads is None or callable(reads) else _tuples(reads)
        self.fwd = fwd
        self.order = order
        self.jvp = jvp
        self.involution = involution
        self.fresh = fresh

    def __call__(self, *args, **static):
        if not any(has_node(a) for a in args):
            return self.plain(*args, **static)
        return self.node(*args, **static)

    def node(self, *args, **static):
        # the node of op(*args), also for plain arguments (as constants)
        if (
            simplify
            and self.involution
            and len(args) == 1
            and not static
            and is_node(args[0])
            and args[0].__dict__.get("_primitive") is self
        ):
            stats["simplified"] += 1
            return args[0]._children[0]
        lift = self.lift
        if isinstance(lift, (list, tuple)):
            children = tuple(l(a) for l, a in zip(lift, args))
        else:
            children = tuple(lift(a) for a in args)
        key = None
        if share and not self.fresh:
            key = (self, tuple(map(id, children)), _static_key(static))
            z = _nodes.get(key)
            if z is not None and z.value is None and z.with_gradient == gradient_flag(children):
                stats["shared"] += 1
                return z
        reads = _tuples(self.reads(len(children))) if callable(self.reads) else self.reads
        # the residual of the last plain forward (shared by the two closures,
        # which must not capture the node: a reference loop)
        cell = {}

        def forward():
            values = [value_of(c) for c in children]
            if self.fwd is None:
                return self.plain(*values, **static)
            value, cell["residual"] = self.fwd(*values, **static)
            return value

        def values_for(indices):
            # the values the flows into the children indices read
            if reads is None:
                return [value_of(c) for c in children]
            r = reads[indices[0]] if len(indices) == 1 else set(j for i in indices for j in reads[i])
            return [value_of(c) if j in r else None for j, c in enumerate(children)]

        def backward(z):
            needed = [i for i, c in enumerate(children) if c.with_gradient]
            if not needed:
                return
            if self.order == 1 and recording():
                raise NotImplementedError(f"{self.name} supports first derivatives only")
            extra = static
            if self.fwd is not None:
                extra = dict(static, residual=cell.pop("residual", None))
            if self.joint_vjp is not None:
                for i, flow in self.joint_vjp(z, needed, *values_for(needed), **extra).items():
                    if flow is not None:
                        accum(children[i], flow, 1)
                return
            flow = z.gradient
            for i in needed:
                r = self.vjp(i, flow, *values_for((i,)), **extra)
                if r is not None:
                    accum(children[i], r, 1)

        # (deferred reference: node.py imports the foundation, which defines
        # primitives)
        z = g.ad.reverse.node_base(
            forward,
            backward,
            children,
            _container=self.container(*[c._container for c in children], **static),
            _tag=self.name,
        )
        z._reads_children = reads
        # (the static arguments, for printing: see node_str)
        z._static = static
        z._primitive = self
        if key is not None:
            z._key = key
            _nodes[key] = z
        # (the vjp sees the values of the children, never the node's own)
        z._reads_self = False
        if self.jvp is not None and not static:
            z._jvp = self.jvp
        elif self.jvp is not None:
            jvp = self.jvp
            # (static only: the rule must not capture the node)
            z._jvp = lambda z, children, tangents: jvp(z, children, tangents, **static)
        return z
