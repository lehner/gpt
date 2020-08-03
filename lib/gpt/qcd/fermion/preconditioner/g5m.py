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
from gpt.params import params_convention

# (G5 M G5 M)^-1 G5 M G5 = M^-1 G5 M^-1 G5^2 M G5 = M^-1
#
# Mpc = G5M G5M
# L = 1
# R = G5 M G5
# S = 0
#
class g5m_ne_instance:
    def __init__(self, matrix):

        self.F_grid = matrix.F_grid
        ftmp = gpt.vspincolor(self.F_grid)

        def _Mpc(dst, src):
            matrix.G5M.mat(ftmp, src)
            matrix.G5M.mat(dst, ftmp)

        def _L(dst, src):
            gpt.copy(dst, src)

        def _R(dst, src):
            dst @= matrix.G5M * gpt.gamma[5] * src

        def _S(dst, src):
            dst[:] = 0

        self.Mpc = gpt.matrix_operator(mat=_Mpc, otype=matrix.otype, grid=self.F_grid)
        self.L = gpt.matrix_operator(mat=_L, otype=matrix.otype, grid=self.F_grid)
        self.R = gpt.matrix_operator(mat=_R, otype=matrix.otype, grid=self.F_grid)
        self.S = gpt.matrix_operator(mat=_S, otype=matrix.otype, grid=self.F_grid)


class g5m_ne:
    @params_convention()
    def __init__(self, params):
        self.params = params

    def __call__(self, op):
        return g5m_ne_instance(op)
