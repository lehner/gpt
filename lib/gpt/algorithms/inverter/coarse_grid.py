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
import gpt as g
from gpt.algorithms import base


class coarse_grid(base):
    def __init__(self, coarse_inverter, block_map, coarsen_matrix):
        super().__init__()
        self.coarse_inverter = coarse_inverter
        self.block_map = block_map
        self.coarsen_matrix = coarsen_matrix

    def __call__(self, mat):
        assert isinstance(mat, g.matrix_operator)
        vector_space = mat.vector_space

        cmat = self.coarsen_matrix(self.block_map, mat)

        cinv = self.coarse_inverter(cmat)

        @self.timed_function
        def inv(dst, src, t):
            assert dst != src
            self.log(f"{self.block_map.coarse_grid.fdimensions}" + " {")

            t("project")
            src_c = self.block_map.project(src)
            t("coarse inverter")
            dst_c = cinv(src_c)
            del src_c
            t("promote")
            self.block_map.promote(dst, dst_c)
            t()

            self.log("}")

        return g.matrix_operator(
            mat=inv,
            inv_mat=mat,
            adj_mat=None,
            adj_inv_mat=None,
            vector_space=vector_space,
            accept_guess=(True, False),
            accept_list=True,
        )
