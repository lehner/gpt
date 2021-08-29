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

# M^-1 = L Mpc^-1 R + S
class preconditioned(base):
    @g.params_convention()
    def __init__(self, preconditioner, inverter, params):
        super().__init__()
        self.params = params
        self.preconditioner = preconditioner
        self.inverter = inverter

    def __call__(self, mat):
        matrix = self.preconditioner(mat)
        inv_mat = self.inverter(matrix.Mpc)

        @self.timed_function
        def inv(dst, src, t):
            t("prepare")
            pc_src = g(matrix.R * src)
            pc_dst = g(matrix.L.inv() * dst)
            t("inv mat")
            inv_mat(pc_dst, pc_src)
            t("combine")
            # TODO: further improve this, maybe understand why eval is not optimal
            tmp = g.lattice(dst[0])
            # g.eval(dst, matrix.L * pc_dst + matrix.S * src)
            for i in range(len(dst)):
                matrix.L.mat(dst[i], pc_dst[i])
                matrix.S.mat(tmp, src[i])
                g.axpy(dst[i], 1.0, dst[i], tmp)

        return g.matrix_operator(
            mat=inv,
            inv_mat=mat,
            adj_inv_mat=mat.adj(),
            adj_mat=None,  # implement adj_mat when needed
            otype=mat.otype,
            accept_guess=(True, False),
            grid=mat.grid,
            cb=None,
            accept_list=True,
        )
