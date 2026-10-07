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
import gpt as g
from gpt.ad.reverse.util import container
from gpt.ad.reverse.functional_node import joint_node


def preimage(dfm, fields, indices, inverse, inverter=None):
    """The preimage x = phi^-1(y) of a map phi (a diffeomorphism with
    jacobian) as nodes.  phi updates the fields at indices and keeps the others
    (e.g. its parameters) fixed; fields = y (nodes; plain values are
    constants), inverse(values) returns the preimage of plain values (all
    fields).  Returns one node per index (the others pass through
    unchanged).

    The backward follows from phi(x) = y (first order only): for the flow c
    into the outputs, J_xx^T lambda = c (J the Jacobian of phi at x, J_xx its
    block of the updated fields; solved with inverter), the flow into the
    inputs at indices is lambda, the flow into the other inputs is
    -(d phi / d others)^T lambda.  Both products are one dfm.jacobian call
    (the vector-Jacobian product in the cartesian representation)."""
    if inverter is None:
        inverter = g.algorithms.inverter.fgcr(eps=1e-12, maxiter=1000, restartlen=30)
    state = {}

    def forward(y):
        x = inverse(y)
        state["y"], state["x"] = y, x
        out = [x[i] for i in indices]
        return out[0] if len(indices) == 1 else out

    def jacobian(lam):
        # the vector-Jacobian product of phi at x for a cotangent lam on the
        # updated fields (zero on the others)
        x, y = state["x"], state["y"]
        d = [g.group.zero(v) for v in x]
        for i, l in zip(indices, lam):
            d[i] = l
        return dfm.jacobian(x, y, d)

    def flows(values, z, needed):
        x, y = state["x"], state["y"]
        c = z.gradient if len(indices) > 1 else [z.gradient]
        c = [
            g.group.zero(x[i]) if ci is None else g.infinitesimal_to_cartesian(x[i], ci)
            for i, ci in zip(indices, c)
        ]
        if len(indices) == 1:

            def mat(dst, src):
                dst @= jacobian([src])[indices[0]]

            lam = [inverter(mat)(c[0])]
        else:

            def mat(dst, src):
                lam = g.separate(src, dimension=0)
                dst @= g.merge([jacobian(lam)[i] for i in indices], dimension=0)

            lam = g.separate(inverter(mat)(g.merge(c, dimension=0)), dimension=0)
        jl = jacobian(lam)
        result = {}
        for j in needed:
            if j in indices:
                flow = lam[indices.index(j)]
            else:
                flow = g(-1.0 * jl[j]) if isinstance(jl[j], g.lattice) else -jl[j]
            result[j] = g.cartesian_to_infinitesimal(y[j], flow)
        return result

    def z_container(containers):
        c = containers[indices[0]]
        return c if len(indices) == 1 else container(list, c, len(indices))

    z = joint_node(fields, forward, flows, z_container, "preimage")
    if len(indices) == 1:
        return [z]
    return [z[k] for k in range(len(indices))]
