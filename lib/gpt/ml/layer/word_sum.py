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
# Site-local sums of products of matrix fields as one compiled stencil (all
# factors at the zero point):
#
#   y(x) = sum_t w_t X_{t,1}(x) X_{t,2}(x) ... ,   X = an input or its adjoint
#
# On nodes the stencil is a stencil node (g.ad.reverse.foundation.stencil):
# one kernel forward, its adjoint (again a stencil) backward, and stencils at
# any nesting depth -- instead of a node graph with one node per product and
# sum.  Site-dependent or trained coefficients enter as factor fields c 1
# (layer.util.embed).
#
import gpt as g
from gpt.ad.reverse.util import is_node, value_depth_static


class word_sum:
    # terms: [(weight, [(input index, adjoint), ...]), ...]; the first term
    # starts the output
    def __init__(self, terms):
        self.terms = terms
        self.stencils = {}

    def stencil(self, grid, otype):
        key = (grid, otype.__name__)
        if key not in self.stencils:
            zero = (0,) * grid.nd
            code = [
                (0, -1 if t == 0 else 0, complex(w), [(i + 1, 0, int(a)) for i, a in factors])
                for t, (w, factors) in enumerate(self.terms)
            ]
            self.stencils[key] = g.stencil.matrix(g.lattice(grid, otype), [zero], code)
        return self.stencils[key]

    def __call__(self, inputs):
        nodes = [x for x in inputs if is_node(x)]
        if not nodes:
            inputs = [g(x) for x in inputs]
            out = g.lattice(inputs[0])
            self.stencil(out.grid, out.otype)(out, *inputs)
            return out
        # a zero output at the depth of the deepest input
        out = max(nodes, key=value_depth_static).new()
        self.stencil(out.grid, out.otype)(out, *inputs)
        return out
