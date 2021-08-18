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

#
# M^-1 = L Mpc^-1 R + S
#
# ->
#
# M^-1 = L V^-1 V Mpc^-1 V^-1 V R + S
#      = L_st Mpc_st^-1 R_st + S
#
# with
#
# L_st   = L V^-1
# Mpc_st = V Mpc V^-1
# R_st   = V R
#
# We assume H^dag = H
#
class similarity_transformation:
    def __init__(self, pc, V):

        self.S = pc.S

        def wrap(Mpc, L, R):
            tmp = gpt.lattice(Mpc.grid[0], Mpc.otype[0])

            def _Mpc(o_d, i_d):
                V.inv_mat(o_d, i_d)
                Mpc.mat(tmp, o_d)
                V.mat(o_d, tmp)

            def _Mpc_dag(o_d, i_d):
                V.adj_mat(o_d, i_d)
                Mpc.adj_mat(tmp, o_d)
                V.adj_inv_mat(o_d, tmp)

            def _R(o_d, i):
                R.mat(tmp, i)
                V.mat(o_d, tmp)

            def _R_dag(o, i_d):
                V.adj_mat(tmp, i_d)
                R.adj_mat(o, tmp)

            def _L(o, i_d):
                V.inv_mat(tmp, i_d)
                L.mat(o, tmp)

            def _L_inv(o_d, i):
                L.inv_mat(tmp, i)
                V.mat(o_d, tmp)

            wrapped_R = gpt.matrix_operator(
                mat=_R,
                adj_mat=_R_dag,
                otype=R.otype,
                grid=R.grid,
                cb=R.cb,
            )

            wrapped_L = gpt.matrix_operator(
                mat=_L,
                inv_mat=_L_inv,
                otype=L.otype,
                grid=L.grid,
                cb=L.cb,
            )

            wrapped_Mpc = gpt.matrix_operator(
                mat=_Mpc, adj_mat=_Mpc_dag, otype=Mpc.otype, grid=Mpc.grid, cb=Mpc.cb
            )

            return wrapped_Mpc, wrapped_L, wrapped_R

        self.Mpc, self.L, self.R = wrap(pc.Mpc, pc.L, pc.R)

        self.Mpc.split = lambda mpi: wrap(pc.Mpc.split(mpi), pc.L, pc.R)[0]
