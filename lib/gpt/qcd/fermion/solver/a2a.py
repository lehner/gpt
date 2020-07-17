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


class a2a:
    def __init__(self, matrix):
        self.matrix = matrix
        self.F_grid_eo = matrix.F_grid_eo
        self.F_grid = matrix.F_grid
        self.U_grid = matrix.U_grid
        self.otype = matrix.otype

        self.oe = gpt.lattice(self.F_grid_eo, self.otype)
        self.oo = gpt.lattice(self.F_grid_eo, self.otype)
        self.U_tmp = gpt.lattice(self.U_grid, self.otype)
        self.F_tmp = gpt.lattice(self.F_grid, self.otype)
        self.F_tmp_2 = gpt.lattice(self.F_grid, self.otype)

        def _v_unphysical(dst, evec):
            self.matrix.L.mat(dst, evec)

        def _w_unphysical(dst, evec):
            self.matrix.R.adj_mat(dst, evec)

        def _v(dst, evec):
            _v_unphysical(self.F_tmp, evec)
            self.matrix.ExportPhysicalFermionSolution(dst, self.F_tmp)

        def _w(dst, evec):
            _w_unphysical(self.F_tmp, evec)
            self.matrix.Dminus.adj_mat(self.F_tmp_2, self.F_tmp)
            self.matrix.ExportPhysicalFermionSource(dst, self.F_tmp_2)

        def _G5w(dst, evec):
            _w(self.U_tmp, evec)
            dst @= gpt.gamma[5] * self.U_tmp

        self.v = gpt.matrix_operator(
            mat=_v,
            otype=self.otype,
            zero=(False, False),
            grid=(self.U_grid, self.F_grid_eo),
        )

        self.w = gpt.matrix_operator(
            mat=_w,
            otype=self.otype,
            zero=(False, False),
            grid=(self.U_grid, self.F_grid_eo),
        )

        self.G5w = gpt.matrix_operator(
            mat=_G5w,
            otype=self.otype,
            zero=(False, False),
            grid=(self.U_grid, self.F_grid_eo),
        )

        self.v_unphysical = gpt.matrix_operator(
            mat=_v_unphysical,
            otype=self.otype,
            zero=(False, False),
            grid=(self.F_grid, self.F_grid_eo),
        )

        self.w_unphysical = gpt.matrix_operator(
            mat=_w_unphysical,
            otype=self.otype,
            zero=(False, False),
            grid=(self.F_grid, self.F_grid_eo),
        )
