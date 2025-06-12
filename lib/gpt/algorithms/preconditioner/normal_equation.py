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
# M^-1 = L (Mpc^dag Mpc)^-1 Mpc^dag R + S
#      = L Mpc_ne^-1 R_ne + S
#
# with
#
# Mpc_ne = Mpc^dag Mpc
# R_ne   = Mpc^dag R
#
class normal_equation:
    def __init__(self, pc):
        self.L = pc.L
        self.S = pc.S

        def wrap(Mpc, R):
            Mpc_adj = Mpc.adj()
            R_adj = R.adj()

            def _N_dag_N(o_d, i_d):
                gpt.eval(o_d, Mpc_adj * Mpc * gpt.expr(i_d))

            def _R(o_d, i):
                gpt.eval(o_d, Mpc_adj * R * gpt.expr(i))

            def _R_dag(o, i_d):
                gpt.eval(o, R_adj * Mpc * gpt.expr(i_d))

            wrapped_R = gpt.matrix_operator(
                mat=_R, adj_mat=_R_dag, vector_space=R.vector_space, accept_list=True
            )

            wrapped_Mpc = gpt.matrix_operator(
                mat=_N_dag_N, adj_mat=_N_dag_N, vector_space=Mpc.vector_space, accept_list=True
            )

            return wrapped_Mpc, wrapped_R

        self.Mpc, self.R = wrap(pc.Mpc, pc.R)

        self.Mpc.inherit(pc.Mpc, lambda nop: wrap(nop, pc.R)[0])
