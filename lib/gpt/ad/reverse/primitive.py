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
#                                            flow), or
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
#
# The static keyword arguments are not differentiated; they are passed on
# to plain, container, fwd, the vjp, and to the nodes the vjp builds.
#
import gpt as g
from gpt.ad.reverse.util import constant, is_node, value_of, recording
from gpt.ad.reverse.flow import accum


def has_node(x):
    # a node, or a (nested) list containing one
    if isinstance(x, (list, tuple)):
        return any(has_node(y) for y in x)
    return is_node(x)


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
    ):
        assert (vjp is None) != (joint_vjp is None), "a primitive has a vjp or a joint_vjp"
        self.name = name
        self.plain = plain
        self.container = container
        self.vjp = vjp
        self.joint_vjp = joint_vjp
        self.lift = lift
        self.reads = reads
        self.fwd = fwd
        self.order = order
        self.jvp = jvp

    def __call__(self, *args, **static):
        if not any(has_node(a) for a in args):
            return self.plain(*args, **static)
        return self.node(*args, **static)

    def node(self, *args, **static):
        # the node of op(*args), also for plain arguments (as constants)
        lift = self.lift if isinstance(self.lift, (list, tuple)) else [self.lift] * len(args)
        children = tuple(l(a) for l, a in zip(lift, args))
        reads = self.reads(len(children)) if callable(self.reads) else self.reads
        if reads is not None:
            reads = tuple(tuple(r) for r in reads)
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
            r = set(j for i in indices for j in reads[i])
            return [value_of(c) if j in r else None for j, c in enumerate(children)]

        def backward(z):
            needed = [i for i, c in enumerate(children) if c.with_gradient]
            if not needed:
                return
            if self.order == 1 and recording():
                raise NotImplementedError(f"{self.name} supports first derivatives only")
            extra = {}
            if self.fwd is not None:
                extra["residual"] = cell.pop("residual", None)
            if self.joint_vjp is not None:
                flows = self.joint_vjp(z, needed, *values_for(needed), **extra, **static)
            else:
                flows = {
                    i: self.vjp(i, z.gradient, *values_for([i]), **extra, **static)
                    for i in needed
                }
            for i, flow in flows.items():
                if flow is not None:
                    accum(children[i], flow, 1)

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
        # (the vjp sees the values of the children, never the node's own)
        z._reads_self = False
        if self.jvp is not None:
            jvp = self.jvp
            # (static only: the rule must not capture the node)
            z._jvp = lambda z, children, tangents: jvp(z, children, tangents, **static)
        return z
