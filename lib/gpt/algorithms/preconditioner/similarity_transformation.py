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

        V_inv = V.inv()
        V_adj = V.adj()
        V_adj_inv = V_adj.inv()

        def wrap(Mpc, L, R):
            Mpc_adj = Mpc.adj()
            R_adj = R.adj()
            L_inv = L.inv()

            def _Mpc(o_d, i_d):
                gpt.eval(o_d, V * Mpc * V_inv * gpt.expr(i_d))

            def _Mpc_dag(o_d, i_d):
                gpt.eval(o_d, V_adj_inv * Mpc_adj * V_adj * gpt.expr(i_d))

            def _R(o_d, i):
                gpt.eval(o_d, V * R * gpt.expr(i))

            def _R_dag(o, i_d):
                gpt.eval(o, R_adj * V_adj * gpt.expr(i_d))

            def _L(o, i_d):
                gpt.eval(o, L * V_inv * gpt.expr(i_d))

            def _L_inv(o_d, i):
                gpt.eval(o_d, V * L_inv * gpt.expr(i))

            wrapped_R = gpt.matrix_operator(
                mat=_R, adj_mat=_R_dag, vector_space=R.vector_space, accept_list=True
            )

            wrapped_L = gpt.matrix_operator(
                mat=_L, inv_mat=_L_inv, vector_space=L.vector_space, accept_list=True
            )

            wrapped_Mpc = gpt.matrix_operator(
                mat=_Mpc, adj_mat=_Mpc_dag, vector_space=Mpc.vector_space, accept_list=True
            )

            return wrapped_Mpc, wrapped_L, wrapped_R

        self.Mpc, self.L, self.R = wrap(pc.Mpc, pc.L, pc.R)

        self.Mpc.split = lambda mpi: wrap(pc.Mpc.split(mpi), pc.L, pc.R)[0]
