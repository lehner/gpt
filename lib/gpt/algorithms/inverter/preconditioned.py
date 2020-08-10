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

# M^-1 = L Mpc^-1 R + S
class preconditioned:
    @g.params_convention()
    def __init__(self, preconditioner, inverter, params):
        self.params = params
        self.preconditioner = preconditioner
        self.inverter = inverter

    def __call__(self, mat):

        matrix = self.preconditioner(mat)
        inv_mat = self.inverter(matrix.Mpc)

        def inv(dst, src):
            # TODO: forward guess dst to inverter, need pseudo L.inv(), pick even sites then L.inv() may be EE, do for each PC
            g.eval(dst, matrix.L * inv_mat * matrix.R * src + matrix.S * src)

        return g.matrix_operator(
            mat=inv,
            inv_mat=mat,
            adj_inv_mat=mat.adj(),
            adj_mat=None,  # implement adj_mat when needed
            otype=mat.otype,
            zero=(True, False),
            grid=matrix.F_grid,
            cb=None,
            accept_list=True,
        )
