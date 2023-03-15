#
#    GPT - Grid Python Toolkit
#    Copyright (C) 2023  Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)
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
import gpt, cgpt


class transfer:
    def __init__(self, fine_grid, coarse_grid, otype):
        mask = gpt.complex(fine_grid)
        mask[:] = 1

        template = gpt.lattice(fine_grid, otype)
        basis_size = 4  # a known basis size to cgpt

        self.fine_grid = fine_grid
        self.coarse_grid = coarse_grid

        self.obj = cgpt.create_block_map(
            coarse_grid.obj,
            [template] * basis_size,
            basis_size,
            basis_size,
            mask.v_obj[0],
        )

        def _sum(coarse, fine):
            assert len(fine) == len(coarse)
            cgpt.block_sum(self.obj, coarse, fine)
            for i in range(len(fine)):
                coarse[i].checkerboard(fine[i].checkerboard())

        def _embed(fine, coarse):
            assert len(fine) == len(coarse)
            cgpt.block_embed(self.obj, coarse, fine)
            for i in range(len(fine)):
                fine[i].checkerboard(coarse[i].checkerboard())

        self.sum = gpt.matrix_operator(
            mat=_sum,
            adj_mat=_embed,
            vector_space=(
                gpt.vector_space.explicit_grid_otype(coarse_grid, otype),
                gpt.vector_space.explicit_grid_otype(fine_grid, otype),
            ),
            accept_list=True,
        )

        self.embed = gpt.matrix_operator(
            mat=_embed,
            adj_mat=_sum,
            vector_space=(
                gpt.vector_space.explicit_grid_otype(fine_grid, otype),
                gpt.vector_space.explicit_grid_otype(coarse_grid, otype),
            ),
            accept_list=True,
        )

    def __del__(self):
        cgpt.delete_block_map(self.obj)
