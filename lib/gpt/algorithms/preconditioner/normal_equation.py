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
            tmp = gpt.lattice(Mpc.grid[0], Mpc.otype[0])

            def _N_dag_N(o_d, i_d):
                Mpc.mat(tmp, i_d)
                Mpc.adj_mat(o_d, tmp)

            def _R(o_d, i):
                R.mat(tmp, i)
                Mpc.adj_mat(o_d, tmp)

            def _R_dag(o, i_d):
                Mpc.mat(tmp, i_d)
                R.adj_mat(o, tmp)

            wrapped_R = gpt.matrix_operator(
                mat=_R,
                adj_mat=_R_dag,
                otype=R.otype,
                grid=R.grid,
                cb=R.cb,
            )

            wrapped_Mpc = gpt.matrix_operator(
                mat=_N_dag_N,
                adj_mat=_N_dag_N,
                otype=Mpc.otype,
                grid=Mpc.grid,
                cb=Mpc.cb,
            )

            return wrapped_Mpc, wrapped_R

        self.Mpc, self.R = wrap(pc.Mpc, pc.R)

        self.Mpc.split = lambda mpi: wrap(pc.Mpc.split(mpi), pc.R)[0]
