#
#    GPT - Grid Python Toolkit
#    Copyright (C) 2020  Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)
#                  2020  Daniel Richtmann (daniel.richtmann@ur.de)
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


class multi_grid_setup:
    @g.params_convention(block_size=None, projector=None)
    def __init__(self, params):
        self.block_size = params["block_size"]
        self.n = len(self.block_size)
        self.projector = g.util.to_list(params["projector"]) * self.n

    def __call__(self, matrix):
        levels = []
        grid = matrix.vector_space[0].grid
        for i in range(self.n):
            grid = g.block.grid(grid, self.block_size[i])
            basis = self.projector[i](matrix, grid)
            levels.append((grid, basis))
            if i != self.n - 1:
                matrix = matrix.coarsened(*levels[-1])
        return levels


class coarse_grid(base):
    @g.params_convention(make_hermitian=False, save_links=True)
    def __init__(self, coarse_inverter, coarse_grid, basis, params):
        super().__init__()
        self.params = params
        self.coarse_inverter = coarse_inverter
        self.coarse_grid = coarse_grid
        self.basis = basis

    def __call__(self, mat):
        assert isinstance(mat, g.matrix_operator)
        vector_space = mat.vector_space

        bm = g.block.map(self.coarse_grid, self.basis)

        cmat = mat.coarsened(self.coarse_grid, self.basis)

        cinv = self.coarse_inverter(cmat)

        @self.timed_function
        def inv(dst, src, t):
            assert dst != src
            self.log(f"{self.coarse_grid.fdimensions}" + " {")

            t("project")
            src_c = bm.project(src)
            t("coarse inverter")
            dst_c = cinv(src_c)
            del src_c
            t("promote")
            bm.promote(dst, dst_c)
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
