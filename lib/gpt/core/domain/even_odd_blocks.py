#
#    GPT - Grid Python Toolkit
#    Copyright (C) 2020  Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)
#                  2020  Mattia Bruno
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
import gpt, cgpt, numpy

# ix = x[0] + lat[0]*(x[1] + lat[2]*...)
def index_to_coordinate(ix, lat):  # go through cgpt and order=lexicographic
    x = [0] * len(lat)
    for mu in range(len(lat)):
        x[mu] = ix % lat[mu]
        ix //= lat[mu]
    return x


class even_odd_blocks:
    def __init__(self, grid, block_size, parity):
        self.parity = parity
        self.block_size = block_size
        self.block_volume = int(numpy.prod(block_size))
        self.grid = grid
        assert grid.cb.n == 1  # for now only full grids are supported
        self.promote_plan = {}
        self.project_plan = {}

        # general blocking info
        nd = len(block_size)
        assert nd == grid.nd
        assert numpy.all(numpy.mod(grid.ldimensions, block_size) == 0)
        self.local_blocks_per_dimension = [
            int(d // b) for d, b in zip(grid.ldimensions, block_size)
        ]
        self.blocks_per_dimension = [
            int(d // b) for d, b in zip(grid.gdimensions, block_size)
        ]
        self.number_of_local_blocks = int(numpy.prod(self.local_blocks_per_dimension))
        assert self.number_of_local_blocks % 2 == 0

        # find checkerboarding dimension
        self.extended_local_blocks_per_dimension = [1] * nd
        for mu in reversed(range(nd)):
            if self.local_blocks_per_dimension[mu] > 1:
                self.extended_local_blocks_per_dimension[mu] = (
                    self.number_of_local_blocks // 2
                )
                break
        extended_block_size = [
            block_size[mu] * self.extended_local_blocks_per_dimension[mu]
            for mu in range(nd)
        ]

        # block grid local to the node
        self.local_grid = grid.split([1] * nd, extended_block_size)

        # map local positions in one-dimensional stack of blocks
        self.lcoor = numpy.zeros((self.local_grid.gsites, nd), dtype=numpy.int32)
        offset = [0] * nd
        for mu in range(nd):
            if self.extended_local_blocks_per_dimension[mu] > 1:
                offset[mu] = 1
        for n in range(self.number_of_local_blocks // 2):
            sl = slice(n * self.block_volume, (n + 1) * self.block_volume)
            top = [offset[mu] * n * self.block_size[mu] for mu in range(nd)]
            bottom = [top[mu] + self.block_size[mu] for mu in range(nd)]
            _pos = cgpt.coordinates_from_cartesian_view(
                top, bottom, self.local_grid.cb.cb_mask, None, "lexicographic"
            )
            self.lcoor[sl, :] = _pos

        # map corresponding global positions
        self.gcoor = numpy.zeros((self.local_grid.gsites, nd), dtype=numpy.int32)
        n = 0
        offset = [grid.processor_coor[mu] * grid.ldimensions[mu] for mu in range(nd)]
        for ib in range(self.number_of_local_blocks):
            block_coordinate = index_to_coordinate(ib, self.local_blocks_per_dimension)
            _eo = int(numpy.sum(block_coordinate) % 2)

            if _eo == parity.tag:
                sl = slice(n * self.block_volume, (n + 1) * self.block_volume)
                n += 1

                top = [
                    offset[mu] + block_coordinate[mu] * block_size[mu]
                    for mu in range(len(offset))
                ]
                bottom = [top[mu] + block_size[mu] for mu in range(len(offset))]
                pos = cgpt.coordinates_from_cartesian_view(
                    top, bottom, grid.cb.cb_mask, None, "lexicographic"
                )
                self.gcoor[sl, :] = pos
        assert n * 2 == self.number_of_local_blocks

    def lattice(self, otype):
        return gpt.lattice(self.local_grid, otype)

    def project(self, dst, src):
        tag = src.otype.__name__
        if tag not in self.project_plan:
            plan = gpt.copy_plan(dst, src, embed_in_communicator=src.grid)
            plan.destination += dst.view[self.lcoor]
            plan.source += src.view[self.gcoor]
            self.project_plan[tag] = plan()
        self.project_plan[tag](dst, src)

    def promote(self, dst, src):
        tag = src.otype.__name__
        if tag not in self.promote_plan:
            plan = gpt.copy_plan(dst, src, embed_in_communicator=dst.grid)
            plan.destination += dst.view[self.gcoor]
            plan.source += src.view[self.lcoor]
            self.promote_plan[tag] = plan()
        self.promote_plan[tag](dst, src)
