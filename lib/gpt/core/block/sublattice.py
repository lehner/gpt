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
# Linear maps between a fine grid and the coarse grid of its blocks of
# b_mu sites in direction mu (any otype, full grids):
#
#   restrict(o)   y(c) = x(b c + o)                      (strided pick)
#   prolong(o)    x(b c + o) = y(c), zero elsewhere      (its adjoint)
#   sum           y(c) = sum_{o in block} x(b c + o)     (block sum)
#   broadcast     x(b c + o) = y(c)                      (its adjoint)
#
# All are g.matrix_operators with their adjoints, so they act on reverse-AD
# lattice nodes to any order (A(x) and A * x).  restrict / prolong are cached
# copy plans, which pull each coarse site from the rank holding the fine
# site (any rank layouts of the two grids); sum / broadcast are cgpt's
# block_sum / block_embed, which are rank-local, so a coarse grid in another
# rank layout is served through a coarse grid in the fine layout and a
# repartition.
#
import gpt
import numpy as np


def _paired(a, b):
    # a.adj() is b and b.adj() is a (instead of a new operator per call), so
    # the nodes the vjps build are shared between consumers
    a.adj = lambda: b
    b.adj = lambda: a
    return a, b


class sublattice:
    def __init__(self, fine_grid, factor, coarse_grid=None):
        nd = fine_grid.nd
        factor = [factor] * nd if isinstance(factor, int) else list(factor)
        assert fine_grid.cb.n == 1 and len(factor) == nd
        if coarse_grid is None:
            coarse_grid = gpt.block.grid(fine_grid, factor)
        assert coarse_grid.fdimensions == [f // b for f, b in zip(fine_grid.fdimensions, factor)]
        self.fine_grid = fine_grid
        self.coarse_grid = coarse_grid
        self.factor = factor
        self.local_coarse_grid = coarse_grid
        if list(coarse_grid.mpi) != list(fine_grid.mpi):
            self.local_coarse_grid = gpt.grid(
                coarse_grid.fdimensions, fine_grid.precision, mpi=fine_grid.mpi
            )
        self.plans = {}
        self.transfers = {}
        self.operators = {}

        vs_coarse = gpt.vector_space.explicit_grid(coarse_grid)
        vs_fine = gpt.vector_space.explicit_grid(fine_grid)

        def _sum(coarse, fine):
            for c, f in zip(coarse, fine):
                if self.local_coarse_grid is coarse_grid:
                    self.transfer(f.otype).sum.mat([c], [f])
                else:
                    t = gpt.lattice(self.local_coarse_grid, f.otype)
                    self.transfer(f.otype).sum.mat([t], [f])
                    self.repartition(c, t)

        def _broadcast(fine, coarse):
            for f, c in zip(fine, coarse):
                if self.local_coarse_grid is not coarse_grid:
                    t = gpt.lattice(self.local_coarse_grid, c.otype)
                    self.repartition(t, c)
                    c = t
                self.transfer(c.otype).embed.mat([f], [c])

        self.sum, self.broadcast = _paired(
            gpt.matrix_operator(
                mat=_sum, adj_mat=_broadcast, vector_space=(vs_coarse, vs_fine), accept_list=True
            ),
            gpt.matrix_operator(
                mat=_broadcast, adj_mat=_sum, vector_space=(vs_fine, vs_coarse), accept_list=True
            ),
        )

    def restrict(self, offset=None):
        return self._operators(offset)[0]

    def prolong(self, offset=None):
        return self._operators(offset)[1]

    def _operators(self, offset):
        offset = (0,) * self.fine_grid.nd if offset is None else tuple(int(o) for o in offset)
        assert all(0 <= o < b for o, b in zip(offset, self.factor))
        if offset not in self.operators:

            def _restrict(coarse, fine):
                for c, f in zip(coarse, fine):
                    self.plan(f.otype, offset, True)(c, f)

            def _prolong(fine, coarse):
                for f, c in zip(fine, coarse):
                    f[:] = 0
                    self.plan(c.otype, offset, False)(f, c)

            vs_coarse = gpt.vector_space.explicit_grid(self.coarse_grid)
            vs_fine = gpt.vector_space.explicit_grid(self.fine_grid)
            self.operators[offset] = _paired(
                gpt.matrix_operator(
                    mat=_restrict,
                    adj_mat=_prolong,
                    vector_space=(vs_coarse, vs_fine),
                    accept_list=True,
                ),
                gpt.matrix_operator(
                    mat=_prolong,
                    adj_mat=_restrict,
                    vector_space=(vs_fine, vs_coarse),
                    accept_list=True,
                ),
            )
        return self.operators[offset]

    def plan(self, otype, offset, restrict):
        key = (otype.__name__, offset, restrict)
        if key not in self.plans:
            fine = gpt.lattice(self.fine_grid, otype)
            coarse = gpt.lattice(self.coarse_grid, otype)
            cc = gpt.coordinates(self.coarse_grid)
            fc = cc * np.array(self.factor, dtype=cc.dtype) + np.array(offset, dtype=cc.dtype)
            fc = np.ascontiguousarray(fc.astype(cc.dtype))
            if restrict:
                plan = gpt.copy_plan(coarse, fine, embed_in_communicator=self.fine_grid)
                plan.destination += coarse.view[cc]
                plan.source += fine.view[fc]
            else:
                plan = gpt.copy_plan(fine, coarse, embed_in_communicator=self.fine_grid)
                plan.destination += fine.view[fc]
                plan.source += coarse.view[cc]
            self.plans[key] = plan()
        return self.plans[key]

    def transfer(self, otype):
        key = otype.__name__
        if key not in self.transfers:
            self.transfers[key] = gpt.block.transfer(self.fine_grid, self.local_coarse_grid, otype)
        return self.transfers[key]

    def repartition(self, dst, src):
        # dst = src between the coarse grid and the coarse grid in the fine
        # rank layout (either way)
        key = (src.otype.__name__, "repartition", dst.grid is self.coarse_grid)
        if key not in self.plans:
            plan = gpt.copy_plan(dst, src, embed_in_communicator=self.fine_grid)
            cc = gpt.coordinates(dst.grid)
            plan.destination += dst.view[cc]
            plan.source += src.view[cc]
            self.plans[key] = plan()
        self.plans[key](dst, src)
