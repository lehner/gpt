#
#    GPT - Grid Python Toolkit
#    Copyright (C) 2020  Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)
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
import gpt, math


class cartesian_view:
    def __init__(self, first, second=None, third=None, fourth=None, fifth=None):
        if isinstance(first, gpt.grid) and first.cb == gpt.full:
            g = first
            rank = g.processor
            fdimensions = g.fdimensions
            g_cb = gpt.full
            l_cb = gpt.none
            if second is None:
                mpi = g.mpi
            else:
                assert 0
        elif isinstance(first, gpt.lattice):
            g = first.grid
            rank = g.processor
            fdimensions = g.fdimensions
            g_cb = g.cb
            l_cb = first.checkerboard()
            if second is None:
                mpi = g.mpi
            else:
                assert 0
        else:
            rank, fdimensions, g_cb, l_cb = first, third, fourth, fifth
            if isinstance(second, str):
                mpi = [int(x) for x in second.strip("[]").split(",")]
            else:
                mpi = second
        assert len(mpi) == len(fdimensions)
        self.nd = len(mpi)

        if isinstance(g_cb, type):
            g_cb = g_cb(self.nd)

        self.rank = rank
        self.mpi = mpi
        self.fdimensions = fdimensions
        self.ranks = 1
        self.processor_coor = [0] * self.nd
        self.view_dimensions = [fdimensions[i] // mpi[i] for i in range(self.nd)]
        self.checker_dim_mask = g_cb.cb_mask
        self.cb = l_cb.tag

        for i in range(self.nd):
            assert fdimensions[i] % mpi[i] == 0
            self.ranks *= mpi[i]
            self.processor_coor[i] = rank % mpi[i]
            rank = rank // mpi[i]

        if self.rank < 0 or self.rank >= self.ranks:
            self.processor_coor = [None] * self.nd
            self.top = [0] * self.nd
            self.bottom = [0] * self.nd
        else:
            self.top = [self.view_dimensions[i] * self.processor_coor[i] for i in range(self.nd)]
            self.bottom = [self.top[i] + self.view_dimensions[i] for i in range(self.nd)]

    def describe(self):
        return str(self.mpi).replace(" ", "")

    def views_for_node(self, grid):
        # need to have same length on each node but can have None entry if node does not participate
        grid_rank = grid.cartesian_rank()
        grid_stride = grid.Nprocessors
        views_per_node = int(math.ceil(self.ranks / grid.Nprocessors))

        ngroups = int(math.ceil(grid.Nprocessors / gpt.default.max_io_nodes))

        # first group
        views = []
        for igroup in range(ngroups):
            for idx in range(views_per_node):
                iview = grid_rank + idx * grid_stride
                if iview % ngroups == igroup and iview < self.ranks:
                    iv = iview
                else:
                    iv = None
                views.append(iv)

        return views
