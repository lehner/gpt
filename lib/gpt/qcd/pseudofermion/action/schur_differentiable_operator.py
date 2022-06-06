#
#    GPT - Grid Python Toolkit
#    Copyright (C) 2020  Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)
#                  2020  Mattia Bruno
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


class MMdag:
    def M(op):
        return op

    def Mdag(op):
        return op.adj()

    def MMdag(op):
        def operator(dst, src):
            dst @= op * g(op.adj() * src)

        return operator

    def Mderiv(op):
        return op.M_projected_gradient

    def MderivDag(op):
        return op.M_projected_gradient.adj()


#      ( EE EO )   ( 1    EO OO^-1 ) ( Mhat 0 ) ( 1  0  )
#  M = ( OE OO ) = ( 0    1        ) ( 0    1 ) ( OE OO )
# Mhat = EE - EO OO^-1 OE
class MMdag_evenodd:
    def M(self, op):
        tmp = op.Mooee.vector_space[0].lattice()

        def operator(dst, src):
            op.Meooe.mat(dst, src)
            op.Mooee.inv_mat(tmp, dst)
            op.Meooe.mat(dst, tmp)
            op.Mooee.mat(tmp, src)
            dst @= tmp - dst

        return g.matrix_operator(mat=operator, vector_space=op.Mooee.vector_space)

    def Mdag(self, op):
        tmp = op.Mooee.vector_space[0].lattice()

        def operator(dst, src):
            op.Meooe.adj_mat(dst, src)
            op.Mooee.adj_inv_mat(tmp, dst)
            op.Meooe.adj_mat(dst, tmp)
            op.Mooee.adj_mat(tmp, src)
            dst @= tmp - dst

        return g.matrix_operator(mat=operator, vector_space=op.Mooee.vector_space)

    def MMdag(self, op):
        def spawn(op):
            tmp = [op.Mooee.vector_space[0].lattice() for _ in [0, 1]]

            def operator(dst, src):
                op.Meooe.adj_mat(dst, src)
                op.Mooee.adj_inv_mat(tmp[0], dst)
                op.Meooe.adj_mat(dst, tmp[0])
                op.Mooee.adj_mat(tmp[0], src)

                tmp[1] @= tmp[0] - dst

                op.Meooe.mat(dst, tmp[1])
                op.Mooee.inv_mat(tmp[0], dst)
                op.Meooe.mat(dst, tmp[0])
                op.Mooee.mat(tmp[0], tmp[1])

                dst @= tmp[0] - dst

            return g.matrix_operator(mat=operator, vector_space=op.Mooee.vector_space).inherit(
                op, lambda nop: spawn(nop)
            )

        return spawn(op)

    def Mderiv(self, op):
        tmp = [op.Mooee.vector_space[0].lattice() for _ in [0, 1]]

        def operator(left, right):
            op.Meooe.mat(tmp[0], right)
            op.Mooee.inv_mat(tmp[1], tmp[0])
            frc_o = op.Meooe_projected_gradient(left, tmp[1])

            op.Meooe.adj_mat(tmp[0], left)
            op.Mooee.adj_inv_mat(tmp[1], tmp[0])
            frc_e = op.Meooe_projected_gradient(tmp[1], right)

            frc = g.group.cartesian(op.U)
            for mu in range(len(frc)):
                g.set_checkerboard(frc[mu], frc_o[mu])
                g.set_checkerboard(frc[mu], frc_e[mu])
                frc[mu] *= -1.0
            return frc

        return operator

    def MderivDag(self, op):
        tmp = [op.Mooee.vector_space[0].lattice() for _ in [0, 1]]

        def operator(left, right):
            op.Meooe.adj_mat(tmp[0], right)
            op.Mooee.adj_inv_mat(tmp[1], tmp[0])
            frc_o = op.Meooe_projected_gradient.adj()(left, tmp[1])

            op.Meooe.mat(tmp[0], left)
            op.Mooee.inv_mat(tmp[1], tmp[0])
            frc_e = op.Meooe_projected_gradient.adj()(tmp[1], right)

            frc = g.group.cartesian(op.U)
            for mu in range(len(frc)):
                g.set_checkerboard(frc[mu], frc_o[mu])
                g.set_checkerboard(frc[mu], frc_e[mu])
                frc[mu] *= -1.0
            return frc

        return operator
