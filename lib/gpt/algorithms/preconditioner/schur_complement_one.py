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

# General block matrix with domain D and its complement C
#
#      ( DD DC )   ( DD   DC CC^-1 ) ( Mpc  0 ) ( 1  0  )
#  M = ( CD CC ) = ( 0    1        ) ( 0    1 ) ( CD CC )
#
#  Mpc = 1 - DD^-1 DC CC^-1 CD   (Schur complement one)
#
# Then
#
#    det(M) = det(DD) det(CC) det(Mpc)
#
# and
#
#        ( 1  0  )^-1 ( Mpc^-1  0 ) ( DD        DC CC^-1  )^-1
# M^-1 = ( CD CC )    ( 0       1 ) ( 0         1         )
#
#        ( 1              0   )  ( Mpc^-1   0 )  ( DD^-1   - DD^-1 DC CC^-1 )
#      = ( -CC^-1 CD    CC^-1 )  ( 0        1 )  ( 0              1         )
#
# M^-1 = L Mpc^-1 R + S
#
# R = DD^-1 ( 1   - DC CC^-1 )  ;  R^dag = ( 1   - DC CC^-1 )^dag DD^-1^dag
#
#     ( 1         )
# L = ( -CC^-1 CD )
#
#     ( 0   0     )
# S = ( 0   CC^-1 )
#
# A2A:
#
# M^-1 = L |n><n| R + S = v w^dag + S ;  -> v = L |n>, w = R^dag |n>
#
class schur_complement_one:
    def __init__(self, op, domain_decomposition):

        otype = op.otype[0]
        op_domain = op.domain

        dd_op = domain_decomposition(op)

        DD = dd_op.DD
        CC = dd_op.CC
        CD = dd_op.CD
        DC = dd_op.DC

        D_domain = dd_op.D_domain
        C_domain = dd_op.C_domain

        tmp_d = [D_domain.lattice(otype) for i in range(2)]
        tmp_c = [C_domain.lattice(otype) for i in range(2)]

        def _N(o_d, i_d):
            CD.mat(tmp_c[0], i_d)
            CC.inv_mat(tmp_c[1], tmp_c[0])
            DC.mat(tmp_d[0], tmp_c[1])
            DD.inv_mat(o_d, tmp_d[0])
            o_d @= i_d - o_d

        def _N_dag(o_d, i_d):
            DD.adj_inv_mat(tmp_d[0], i_d)
            DC.adj_mat(tmp_c[0], tmp_d[0])
            CC.adj_inv_mat(tmp_c[1], tmp_c[0])
            CD.adj_mat(o_d, tmp_c[1])
            o_d @= i_d - o_d

        def _L(o, i_d):
            CD.mat(tmp_c[0], i_d)
            CC.inv_mat(tmp_c[1], tmp_c[0])
            tmp_c[1] @= -tmp_c[1]

            D_domain.promote(o, i_d)
            C_domain.promote(o, tmp_c[1])

        def _L_pseudo_inverse(o_d, i):
            D_domain.project(o_d, i)

        def _S(o, i):
            C_domain.project(tmp_c[0], i)
            CC.inv_mat(tmp_c[1], tmp_c[0])
            C_domain.promote(o, tmp_c[1])

            tmp_d[0][:] = 0
            D_domain.promote(o, tmp_d[0])

        self.L = gpt.matrix_operator(
            mat=_L,
            inv_mat=_L_pseudo_inverse,
            otype=otype,
            grid=(op_domain.grid, D_domain.grid),
            cb=(op_domain.checkerboard, D_domain.checkerboard),
        )

        def _R(o_d, i):
            C_domain.project(tmp_c[0], i)
            D_domain.project(tmp_d[0], i)
            CC.inv_mat(tmp_c[1], tmp_c[0])
            DC.mat(o_d, tmp_c[1])
            tmp_d[0] @= tmp_d[0] - o_d
            DD.inv_mat(o_d, tmp_d[0])

        def _R_dag(o, i_d):
            DD.adj_inv_mat(tmp_d[0], i_d)
            D_domain.promote(o, tmp_d[0])
            DC.adj_mat(tmp_c[0], tmp_d[0])
            tmp_c[0] @= -tmp_c[0]
            CC.adj_inv_mat(tmp_c[1], tmp_c[0])
            C_domain.promote(o, tmp_c[1])

        self.R = gpt.matrix_operator(
            mat=_R,
            adj_mat=_R_dag,
            otype=otype,
            grid=(D_domain.grid, op_domain.grid),
            cb=(D_domain.checkerboard, op_domain.checkerboard),
        )

        self.S = gpt.matrix_operator(
            mat=_S, otype=otype, grid=op_domain.grid, cb=op_domain.checkerboard
        )

        self.Mpc = gpt.matrix_operator(
            mat=_N,
            adj_mat=_N_dag,
            otype=otype,
            grid=D_domain.grid,
            cb=D_domain.checkerboard,
        )

        for undressed in ["Mpc"]:
            self.__dict__[undressed].split = lambda mpi: schur_complement_one(
                op.split(mpi), domain_decomposition
            ).__dict__[undressed]
