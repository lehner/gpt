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
# Differentiable symplectic integrators: fields_prime = md(fields, tau).
#
# The integrators of g.algorithms.integrator.symplectic update fields in
# place.  Here an update is a pure map of a flat list of fields (plain or
# nodes): on nodes md builds the trajectory as a graph, so that
# E(fields_prime).backward() gives the gradients w.r.t. the initial fields,
# including parameters of the actions (further fields of the list, read by
# the forces and never updated).  On plain fields it is a plain run,
# bit-identical to the in-place integrators.
#
# The schemes (leap_frog, OMF2, OMF4, nested time scales) are those of
# g.algorithms.integrator.symplectic: an update is a symplectic_base whose op
# is a pure function (eps, direction, fields) -> fields, so schemes.generic
# builds the same step sequence.
#
# A force maps a list of fields (plain or nodes) to a list of cartesian
# fields.  Two forces of an action S:
#
#   gradient(S, n)                    S accepts nodes: a checkpoint whose
#                                     forward is the plain S.gradient and
#                                     whose backward differentiates a recorded
#                                     gradient of S; any order
#   gradient_finite_difference(S, n)  any differentiable_functional: the
#                                     backward is a finite difference of
#                                     S.gradient; first order
#
import gpt as g
from gpt.algorithms.integrator.symplectic import symplectic_base, schemes
from gpt.ad.reverse.primitive import primitive
from gpt.ad.reverse.checkpoint import checkpoint
from gpt.ad.reverse.util import get_container, is_node, list_container, recorded_gradients


class _update:
    # fields[dst] <- compose(sign * eps * force(fields[src]), fields[dst])
    def __init__(self, dst, force, src, sign):
        self.dst = list(dst)
        self.src = list(src)
        self.force = force
        self.sign = sign

    def __call__(self, eps, direction, fields=None):
        if fields is None:
            raise TypeError("a differentiable update maps fields: md(fields, tau)")
        assert direction == 0
        fields = list(fields)
        frc = self.force([fields[i] for i in self.src])
        assert len(frc) == len(self.dst)
        for i, f in zip(self.dst, frc):
            fields[i] = g(g.group.compose(self.sign * eps * f, fields[i]))
        return fields


def update_p(dst, force, src, tag="P"):
    # fields[dst] <- fields[dst] - eps force(fields[src]); dst, src: indices
    # into the list of fields
    s = symplectic_base(tag)
    s.add((dst, _update(dst, force, src, -1), True, tag), +1, 0)
    return s


def update_q(dst, force, src, tag="Q"):
    # fields[dst] <- exp(eps force(fields[src])) fields[dst]
    s = symplectic_base(tag)
    s.add((dst, _update(dst, force, src, +1), True, tag), +1, 0)
    return s


class integrator(symplectic_base):
    # a scheme of differentiable updates; still a symplectic_base, so it can
    # be nested into another scheme (multiple time scales)
    def __init__(self, s):
        super().__init__(s.__name__)
        self.scheme = s.scheme

    def __call__(self, fields, tau):
        verbose = g.default.is_verbose(self.__name__)
        fields = list(fields)
        n = len(self.scheme)
        for i, (op, step, direction) in enumerate(self.scheme):
            if isinstance(op, int):
                raise Exception("Integrator not completely defined")
            if verbose:
                g.message(f"{self.__name__} on step {i}/{n}: {op[-1]}({step * tau}, {direction})")
            fields = op[1](step * tau, direction, fields)
        return fields


def leap_frog(N, i0, i1):
    return integrator(schemes.leap_frog(N, i0, i1))


def OMF2(N, i0, i1, l1=0.18, l2=0.5):
    return integrator(schemes.OMF2(N, i0, i1, l1, l2))


def OMF4(N, i0, i1):
    return integrator(schemes.OMF4(N, i0, i1))


def _cartesian(c):
    # the container of the force on a field of container c
    c = c.copy()
    c.set_otype(c.get_otype().cartesian())
    return c


def _count(fields, n):
    # the number of fields with a force (n; None: all)
    return len(fields) if n is None else n


def _plain_force(S, n, fields):
    # the plain force of S w.r.t. its first n fields
    return g.util.to_list(S.gradient(fields, fields[: _count(fields, n)]))


def gradient(S, n=None):
    # the force of the action S (a functional of a list of fields accepting
    # nodes) w.r.t. its first n fields (default all); the remaining fields
    # are parameters of S (differentiated through, no force returned)
    def recorded(leaves):
        # leaves of a local pass (infinitesimal_to_cartesian=False): record the
        # gradient and convert it explicitly, so that the conversion is
        # differentiated as well
        x = leaves[: _count(leaves, n)]
        return [
            g.infinitesimal_to_cartesian(xi, gr)
            for xi, gr in zip(x, recorded_gradients(S(leaves), x))
        ]

    def f(fields):
        # the checkpoint's forward (plain) and its local passes (leaves)
        if any(is_node(x) for x in fields):
            return recorded(fields)
        return _plain_force(S, n, fields)

    def force(fields):
        fields = list(fields)
        return checkpoint(
            f, fields, [_cartesian(get_container(x)) for x in fields[: _count(fields, n)]]
        )

    force.recorded = recorded
    return force


def gradient_finite_difference(S, n=None, epsilon=1e-3, **scheme):
    # the force of the action S w.r.t. its first n fields (default all; the
    # remaining fields are parameters of S) for any differentiable_functional
    # whose gradient covers all its fields (e.g. a transformed action with
    # transformation parameters and log dets).  First order only: on nodes it
    # is one primitive node whose forward is the plain S.gradient and whose
    # backward needs no graph of S.  With the flows c into the forces (as
    # cartesian directions v), the flow into any field x (a link or a
    # parameter) is, by the symmetry of second derivatives,
    #
    #   grad_x <v, F> = d/de grad_x S(exp(e v) fields[:n]) at e = 0,
    #
    # i.e. the gradient of g.group.directional_derivative(S, v, along) (a
    # finite difference of S.gradient with step epsilon, error O(epsilon^4)
    # for the default scheme, plus the commutator term for SU(N) links):
    # 4 evaluations of the full gradient of S per force in the backward pass
    # (scheme=...: another scheme of directional_derivative).
    def plain(*fields):
        return _plain_force(S, n, list(fields))

    def container(*containers):
        return list_container([_cartesian(c) for c in containers[: _count(containers, n)]])

    def joint_vjp(z, needed, *fields):
        fields = list(fields)
        m = _count(fields, n)
        flows = z.gradient
        if all(c is None for c in flows):
            return {}
        # the flows (infinitesimal convention) as cartesian directions; no flow
        # is a zero direction
        v = []
        for i, c in enumerate(flows):
            if c is None:
                v.append(g.group.zero(fields[i]))
            else:
                v.append(fields[i].otype.cartesian().infinitesimal_to_cartesian(None, g(c)))
        Q = g.group.directional_derivative(S, v, list(range(m)), epsilon, **scheme)
        gr = Q.gradient(fields, [fields[j] for j in needed])
        return {j: g.cartesian_to_infinitesimal(fields[j], x) for j, x in zip(needed, gr)}

    op = primitive("gradient_finite_difference", plain, container, joint_vjp=joint_vjp, order=1)

    def force(fields):
        fields = list(fields)
        z = op(*fields)
        return [z[k] for k in range(_count(fields, n))]

    return force
