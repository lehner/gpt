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
# Gradient checkpointing (Griewank-Walther): y = checkpoint(f, x) is f(x) as
# ONE node of the graph.  Its forward evaluates f on plain values without
# recording a graph, so the graph keeps only the inputs (the node's children),
# not the intermediates of f.  Its backward re-runs f on fresh leaves with the
# input values and back-propagates the flows through this local graph, which
# is freed afterwards: memory of one f instead of the whole chain, for about one
# extra forward of f per backward.
#
# Self-similar to any order: the flow into the inputs is again a checkpoint,
# of the vjp of f, h(x, c) = (dy/dx)^T c, a local reverse pass over f.  In a
# plain pass it is evaluated at once; in a recorded pass (create_graph) it is
# a lazy checkpoint node of the recorded graph, so higher derivatives stay
# checkpointed.  Called on nodes (the leaves of the local pass of the next
# level), h builds f and records its local reverse pass over them, so the
# vjp of h is the vjp of a recorded gradient, an ordinary local graph.
#
# f maps a list of fields to a list of fields of any types (e.g. the nd links
# of a smoothing step, or the links, momenta and parameters of an MD step: the
# outputs are one list node with one container per element; an input may be
# passed through) and must work on plain fields and on nodes.  Every primitive inside f must support the
# derivative order used (e.g. linear_solve: first order only).  No tangent
# rule: a checkpoint on a path of g.ad.reverse.jacobian raises.
#
# inline_nested: checkpoints called on nodes while a local pass of this one
# runs (h at any order) are not nodes but f inline, so the local graph is
# that of f without nested checkpoints.  A nested checkpoint inside a
# recorded local pass re-runs its f once per level of its own tower (the
# value of its vjp, the replay of that vjp, its own backward): inline, f
# runs once, at the memory of the whole local graph (e.g. a Hessian-vector
# product through a checkpointed gradient whose f contains checkpointed
# smoothing steps).
#
import gpt as g
from gpt.ad.reverse.primitive import primitive
from gpt.ad.reverse.util import (
    contraction,
    get_container,
    is_node,
    list_container,
    recorded_gradients,
)


class _inline_state:
    # the number of running local passes of checkpoints with inline_nested
    depth = 0


def _flow(leaf, c):
    # a leaf's gradient, or a zero of its container if it received none
    gr = leaf.gradient
    return c.zero() if gr is None else gr


def _vjp_function(f, n, needed, outputs, inline_nested):
    # h(x_0, ..., x_{n-1}, c_0, ...) = the flows into x_i (i in needed) for the
    # flows c_k into the outputs y_{outputs[k]} of f
    def h(args):
        if not inline_nested:
            return local_pass(args)
        _inline_state.depth += 1
        try:
            return local_pass(args)
        finally:
            _inline_state.depth -= 1

    def local_pass(args):
        x, seeds = list(args[:n]), list(args[n:])
        if not any(is_node(a) for a in args):
            # plain: one local reverse pass over f on fresh leaves (all with
            # a gradient flag, restricted by wrt, so that a node-mode f of the
            # next level can record gradients w.r.t. any of them)
            leaves = [g.ad.reverse.node(v, infinitesimal_to_cartesian=False) for v in x]
            y = f(leaves)
            contraction(seeds, [y[j] for j in outputs]).backward(wrt=[leaves[i] for i in needed])
            return [_flow(leaves[i], get_container(x[i])) for i in needed]
        # nodes (the leaves of a local pass of the next level): f and its
        # recorded local reverse pass over them
        y = f(x)
        return recorded_gradients(contraction(seeds, [y[j] for j in outputs]), [x[i] for i in needed])

    return h


def _plain(*values, f, containers, inline_nested):
    return list(f(list(values)))


def _container(*children, f, containers, inline_nested):
    return list_container(list(containers))


def _joint_vjp(z, needed, *values, f, containers, inline_nested):
    seeds = z.gradient
    outputs = [j for j, c in enumerate(seeds) if c is not None]
    if not outputs:
        return {}
    h = _vjp_function(f, len(values), needed, outputs, inline_nested)
    flow_containers = [get_container(values[i]) for i in needed]
    r = checkpoint(h, list(values) + [seeds[j] for j in outputs], flow_containers, inline_nested)
    return {i: r[k] for k, i in enumerate(needed)}


_checkpoint = primitive("checkpoint", _plain, _container, joint_vjp=_joint_vjp, fresh=True)


def checkpoint(f, inputs, containers=None, inline_nested=False):
    """f(inputs) (a list of fields of any types) as a checkpoint node:
    the graph keeps the inputs only, the backward re-runs f (see above).
    Plain inputs: f(inputs).  containers: those of the outputs (default: from
    f built on lazy zeros of the inputs' containers, without evaluation).
    inline_nested: checkpoints inside the local passes of this one run
    inline (see above).  Inside such a local pass, this is f(inputs)."""
    inputs = list(inputs)
    if _inline_state.depth > 0 or not any(is_node(x) for x in inputs):
        return list(f(inputs))
    if containers is None:
        proxies = [g.ad.reverse.zero(get_container(x)) for x in inputs]
        containers = [get_container(y) for y in f(proxies)]
    containers = tuple(containers)
    z = _checkpoint.node(*inputs, f=f, containers=containers, inline_nested=inline_nested)
    return [z[k] for k in range(len(containers))]
