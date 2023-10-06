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
import gpt


class physical_instance:
    def __init__(self, matrix, pc):
        self.matrix = matrix
        self.F_grid_eo = matrix.F_grid_eo
        self.F_grid = matrix.F_grid
        self.U_grid = matrix.U_grid
        self.otype = matrix.otype

        self.F_tmp = gpt.lattice(self.F_grid, self.otype)
        self.F_tmp_2 = gpt.lattice(self.F_grid, self.otype)

        def _L(dst, src):
            pc.L.mat(self.F_tmp, src)
            self.matrix.ExportPhysicalFermionSolution(dst, self.F_tmp)

        def _R_adj(dst, src):
            pc.R.adj_mat(self.F_tmp, src)
            self.matrix.Dminus.adj_mat(self.F_tmp_2, self.F_tmp)
            self.matrix.ExportPhysicalFermionSource(dst, self.F_tmp_2)

        self.L = gpt.matrix_operator(
            mat=_L,
            accept_guess=(False, False),
            vector_space=(self.matrix.vector_space_U, pc.L.vector_space[1]),
        )

        self.R = gpt.matrix_operator(
            mat=None,
            adj_mat=_R_adj,
            accept_guess=(False, False),
            vector_space=(pc.R.vector_space[0], self.matrix.vector_space_U),
        )


class physical:
    def __init__(self, pc):
        self.pc = pc

    def __call__(self, mat):
        return physical_instance(mat, self.pc(mat))
