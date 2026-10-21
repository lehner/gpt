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
# Node foundation for site-local programs (g.local_program, see
# core/local_program.py): a program called with nodes is one node per group
# of outputs of one otype (a list node for several outputs: a list node has
# one element container), whose
#   - forward is the program's kernel,
#   - backward runs the adjoint program (program.adjoint(needed): one kernel
#     for the flows of all needed inputs), applied as a program again: with
#     plain flows its kernel, in a recorded pass its nodes, whose backward is
#     the adjoint of the adjoint, and so on (a self-similar tower),
#   - tangent rule (g.ad.reverse.jacobian) runs the tangent program for k
#     directions (program.tangent(fiber, k): one kernel), again a program.
#
import gpt as g
from gpt.ad.reverse.primitive import primitive
from gpt.ad.reverse.util import container, constant
from gpt.ad.reverse import tangent


class _program_op:
    # a program with outputs of one otype as a primitive (inputs -> the
    # output, or a list of the m outputs)
    def __init__(self, program):
        self.program = program
        self.m = len(program.outputs)
        self.listed = self.m > 1
        self.op = primitive(
            f"local_program({len(program.inputs)} inputs, {self.m} outputs)",
            self._plain,
            self._container,
            joint_vjp=self._vjp,
            jvp=self._jvp,
        )

    def _container(self, *children):
        elem = container(g.lattice, children[0].get_grid(), self.program.outputs[0])
        return container(list, elem, self.m) if self.listed else elem

    def _plain(self, *values):
        outs = self.program.plain(*values)
        return outs if self.listed else outs[0]

    def _vjp(self, z, needed, *values):
        # the flows of the needed inputs from the adjoint program (plain
        # flows: its kernel; recorded: its nodes)
        z.materialize_gradient()
        psi = list(z.gradient) if self.listed else [z.gradient]
        flows = self.program.adjoint(needed)(*values, *psi)
        return dict(zip(needed, flows))

    def _jvp(self, z, children, tangents):
        # the k tangents of the outputs from the tangent program
        fiber = tuple(i for i, t in enumerate(tangents) if t is not None)
        k = tangent.count(tangents)
        outs = self.program.tangent(fiber, k)(*children, *[tangents[i][j] for j in range(k) for i in fiber])
        if not self.listed:
            return outs
        return [outs[j * self.m : (j + 1) * self.m] for j in range(k)]


def _op(program):
    # (cached on the program, like the primitives of a stencil)
    op = getattr(program, "_node_op", None)
    if op is None:
        op = _program_op(program)
        program._node_op = op
    return op.op


def local_program(program, *fields):
    children = [constant(f) for f in fields]
    groups = {}
    for o, t in enumerate(program.outputs):
        groups.setdefault(t.__name__, []).append(o)
    groups = list(groups.values())
    outs = [None] * len(program.outputs)
    for group in groups:
        z = _op(program if len(groups) == 1 else program.select(group))(*children)
        if len(group) == 1:
            outs[group[0]] = z
        else:
            for j, o in enumerate(group):
                outs[o] = z[j]
    return outs
