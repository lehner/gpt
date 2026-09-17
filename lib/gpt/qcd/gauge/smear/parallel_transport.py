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
from gpt.qcd.gauge.smear.differentiable import dft_diffeomorphism


class parallel_transport(dft_diffeomorphism):
    def __init__(self, U, description, P0=None, P1=None):
        self.description = description
        self.U = U
        self.P0 = P0
        self.P1 = P1

        assert len(description) == len(U)
        nd = len(U)

        cache = {}

        def ft(xU):
            cache_key = f"{type(xU[0])}"
            if cache_key not in cache:
                # paths that share a direction AND a weight are summed inside
                # the stencil, so the staple sum below costs one multiply per
                # distinct weight per direction rather than one multiply and
                # one add per path -- in the primal graph and in the larger
                # derivative graph built from it
                entries = [y for x in description for y in x]
                keys = [i for i, x in enumerate(description) for y in x]
                cache[cache_key] = g.parallel_transport_weighted(xU, entries, keys)

            pt, group_info = cache[cache_key]
            if P0 is not None:
                xU_P0 = [g(xU[i] * P0[i]) for i in range(nd)]
            else:
                xU_P0 = xU

            sU = list(pt(xU_P0))
            assert len(sU) == len(group_info)

            terms = [[] for _ in range(nd)]
            for k, (i, weight) in enumerate(group_info):
                if weight is None:
                    # numeric weight: already folded into the kernel
                    # coefficient, so sU[k] IS the accumulated staple sum
                    xp = sU[k]
                else:
                    if isinstance(weight, g.ad.reverse.node_base):
                        if not isinstance(sU[k], g.ad.reverse.node_base):
                            weight = weight.value
                    xp = g(weight * sU[k])
                if P1 is not None:
                    xp = g(xp * P1[i])
                terms[i].append(xp)

            sm = [None] * nd
            for i in range(nd):
                if not terms[i]:
                    sm[i] = xU[i]
                    continue
                # never accumulate in place: a folded-weight term IS the
                # stencil's own target field, and mutating it would corrupt
                # the stencil output
                acc = terms[i][0]
                for t in terms[i][1:]:
                    acc = acc + t
                sm[i] = g(
                    g.matrix.exp(g.qcd.gauge.project.traceless_anti_hermitian(acc)) * xU[i]
                )
            return sm

        super().__init__(U, ft)

    def inv(self):
        description = [[(-y[0], y[1]) for y in x] for x in self.description]
        return parallel_transport(self.U, description, self.P0, self.P1)
