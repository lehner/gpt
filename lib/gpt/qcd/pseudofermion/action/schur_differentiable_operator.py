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
        return op.Mderiv
    def MderivDag(op):
        return op.MderivDag

#      ( EE EO )   ( 1    EO OO^-1 ) ( Mhat 0 ) ( 1  0  )
#  M = ( OE OO ) = ( 0    1        ) ( 0    1 ) ( OE OO )
# Mhat = EE - EO OO^-1 OE
class MMdag_evenodd:
    def __init__(self, M, parity=g.odd):
        self.F_grid_eo = M.F_grid_eo
        self.parity = parity
        self.tmp = [g.lattice(M.F_grid_eo, M.otype[0]) for _ in [0,1]]
                
    def M(self, op):
        def operator(dst, src):
            op.Meooe.mat(dst, src)
            op.Mooee.inv_mat(self.tmp[0], dst)
            op.Meooe.mat(dst, self.tmp[0])
            op.Mooee.mat(self.tmp[0], src)
            dst @= self.tmp[0] - dst
        return g.matrix_operator(
            mat=operator, otype=op.otype, grid=self.F_grid_eo, cb=self.parity
        )
    
    def Mdag(self, op):
        def operator(dst, src):
            op.Meooe.adj_mat(dst, src)
            op.Mooee.adj_inv_mat(self.tmp[0], dst)
            op.Meooe.adj_mat(dst, self.tmp[0])
            op.Mooee.adj_mat(self.tmp[0], src)
            dst @= self.tmp[0] - dst
        return g.matrix_operator(
            mat=operator, otype=op.otype, grid=self.F_grid_eo, cb=self.parity
        )
    
    def MMdag(self, op):
        def operator(dst, src):
            op.Meooe.adj_mat(dst, src)
            op.Mooee.adj_inv_mat(self.tmp[0], dst)
            op.Meooe.adj_mat(dst, self.tmp[0])
            op.Mooee.adj_mat(self.tmp[0], src)
            
            self.tmp[1] @= self.tmp[0] - dst
    
            op.Meooe.mat(dst, self.tmp[1])
            op.Mooee.inv_mat(self.tmp[0], dst)
            op.Meooe.mat(dst, self.tmp[0])
            op.Mooee.mat(self.tmp[0], self.tmp[1])
            
            dst @= self.tmp[0] - dst
        return operator

    def Mderiv(self, op):
        def operator(left, right):
            op.Meooe.mat(self.tmp[0], right)
            op.Mooee.inv_mat(self.tmp[1], self.tmp[0])
            frc_o = op.Moederiv(left, self.tmp[1])

            op.Meooe.adj_mat(self.tmp[0], left)
            op.Mooee.adj_inv_mat(self.tmp[1], self.tmp[0])
            frc_e = op.Meoderiv(self.tmp[1], right)
            
            frc = g.group.cartesian(op.U)
            for mu in range(len(frc)):
                g.set_checkerboard(frc[mu], frc_o[mu])
                g.set_checkerboard(frc[mu], frc_e[mu])
                frc[mu] *= -1.0
            return frc
        return operator

    def MderivDag(self, op):
        def operator(left, right):
            op.Meooe.adj_mat(self.tmp[0], right)
            op.Mooee.adj_inv_mat(self.tmp[1], self.tmp[0])
            frc_o = op.MoederivDag(left, self.tmp[1])
            
            op.Meooe.mat(self.tmp[0], left)
            op.Mooee.inv_mat(self.tmp[1], self.tmp[0])
            frc_e = op.MeoderivDag(self.tmp[1], right)
            
            frc = g.group.cartesian(op.U)
            for mu in range(len(frc)):
                g.set_checkerboard(frc[mu], frc_o[mu])
                g.set_checkerboard(frc[mu], frc_e[mu])
                frc[mu] *= -1.0
            return frc
        return operator
