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


fingerprint = gpt.default.get_int("--fingerprint", 0) > 1


# General block matrix with domain D and its complement C
#
#      ( DD DC )   ( 1   DC CC^-1 ) ( Mpc  0 ) ( DD  0  )
#  M = ( CD CC ) = ( 0    1       ) ( 0    1 ) ( CD CC )
#
#  Mpc = 1 - DC CC^-1 CD DD^-1   (Schur complement two)
#
# Then
#
#    det(M) = det(DD) det(CC) det(Mpc)
#
# and
#
#        ( DD 0  )^-1 ( Mpc  0 )^-1 ( 1     DC CC^-1  )^-1
# M^-1 = ( CD CC )    ( 0    1 )    ( 0         1     )
#
#        ( DD^-1              0     )  ( Mpc^-1   0 )  ( 1   - DC CC^-1 )
#      = ( -CC^-1 CD DD^-1    CC^-1 )  ( 0        1 )  ( 0       1      )
#
# M^-1 = L Mpc^-1 R + S
#
# R = ( 1   - DC CC^-1 )  ;  R^dag = ( 1   - DC CC^-1 )^dag
#
#     ( DD^-1           )
# L = ( -CC^-1 CD DD^-1 )
#
#     ( 0   0     )
# S = ( 0   CC^-1 )
#
# A2A:
#
# M^-1 = L |n><n| R + S = v w^dag + S ;  -> v = L |n>, w = R^dag |n>
#
class schur_complement_two:
    def __init__(self, op, domain_decomposition):
        dd_op = domain_decomposition(op)

        DD = dd_op.DD
        CC = dd_op.CC
        CD = dd_op.CD
        DC = dd_op.DC

        CC_inv = CC.inv()
        CC_adj_inv = CC_inv.adj()

        DD_inv = DD.inv()
        # DD_adj_inv = DD_inv.adj()

        # CD_adj = CD.adj()
        DC_adj = DC.adj()

        D_domain = dd_op.D_domain
        C_domain = dd_op.C_domain

        op_vector_space = op.vector_space[0]
        C_vector_space = CC.vector_space[0]
        D_vector_space = DD.vector_space[0]
        C_vector_space = CC.vector_space[0]

        tmp_d = [D_vector_space.lattice() for i in range(2)]
        tmp_c = [C_vector_space.lattice() for i in range(2)]

        def _N(o_d, i_d):
            if fingerprint:
                lll = gpt.fingerprint.log()
            DD.inv_mat(tmp_d[0], i_d)
            CD.mat(tmp_c[0], tmp_d[0])
            CC.inv_mat(tmp_c[1], tmp_c[0])
            DC.mat(o_d, tmp_c[1])
            # o_d @= i_d - o_d
            gpt.axpy(o_d, -1.0, o_d, i_d)
            # gpt.eval(o_d, gpt.expr(i_d) - DC * CC_inv * CD * DD_inv * gpt.expr(i_d))
            if fingerprint:
                lll()


        def _N_dag(o_d, i_d):
            if fingerprint:
                lll = gpt.fingerprint.log()
            DC.adj_mat(tmp_c[0], i_d)
            CC.adj_inv_mat(tmp_c[1], tmp_c[0])
            CD.adj_mat(tmp_d[0], tmp_c[1])
            DD.adj_inv_mat(o_d, tmp_d[0])
            # o_d @= i_d - o_d
            gpt.axpy(o_d, -1.0, o_d, i_d)
            # gpt.eval(o_d, gpt.expr(i_d) - DD_adj_inv * CD_adj * CC_adj_inv * DC_adj * gpt.expr(i_d))
            if fingerprint:
                lll()

        def _L(o, i_d):
            tmp = gpt(DD_inv * gpt.expr(i_d))
            D_domain.promote(o, tmp)
            tmp = gpt(-CC_inv * CD * tmp)
            C_domain.promote(o, tmp)

        def _L_pseudo_inverse(o_d, i):
            D_domain.project(o_d, i)
            gpt.eval(o_d, DD * o_d)

        self.L = gpt.matrix_operator(
            mat=_L,
            inv_mat=_L_pseudo_inverse,
            vector_space=(op_vector_space, D_vector_space),
            accept_list=True,
        )

        def _R(o_d, i):
            gpt.eval(o_d, gpt.expr(D_domain.project(i)) - DC * CC_inv * C_domain.project(i))

        def _R_dag(o, i_d):
            D_domain.promote(o, i_d)
            C_domain.promote(o, gpt(-CC_adj_inv * DC_adj * gpt.expr(i_d)))

        self.R = gpt.matrix_operator(
            mat=_R, adj_mat=_R_dag, vector_space=(D_vector_space, op_vector_space), accept_list=True
        )

        def _S(o, i):
            C_domain.promote(o, gpt(CC_inv * C_domain.project(i)))
            D_domain.promote(o, gpt(0.0 * gpt.expr(D_domain.project(i))))

        self.S = gpt.matrix_operator(
            mat=_S, vector_space=(op_vector_space, op_vector_space), accept_list=True
        )

        self.Mpc = gpt.matrix_operator(
            mat=_N, adj_mat=_N_dag, vector_space=(D_vector_space, D_vector_space), accept_list=False
        ).inherit(op, lambda nop: schur_complement_two(nop, domain_decomposition).Mpc)
