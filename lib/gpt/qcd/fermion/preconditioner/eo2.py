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

# First EO preconditioning (upper triangular left)
#
#      ( EE EO )   ( 1  EO OO^-1 ) ( N 0 ) ( EE 0  )
#  M = ( OE OO ) = ( 0    1      ) ( 0 1 ) ( OE OO )
#
#  N = 1 - EO OO^-1 OE EE^-1
#
#  Verify:  ( N   EO OO^-1  ) ( EE 0  )   ( N EE + EO OO^-1 OE    EO )   ; N EE + EO OO^-1 OE = EE - EO OO^-1 OE + EO OO^-1 OE = EE
#           ( 0   1         ) ( OE OO ) = ( OE                    OO )
#
# Then:
#
#        ( EE 0  )^-1 ( N^dag^-1 (N^dag N)  0 )^-1 ( 1     EO OO^-1  )^-1
# M^-1 = ( OE OO )    ( 0                   1 )    ( 0         1     )
#
#        ( EE^-1              0     )  ( (N^dag N)^-1 N^dag   0 )  ( 1   - EO OO^-1 )
#      = ( -OO^-1 OE EE^-1    OO^-1 )  ( 0                    1 )  ( 0       1      )
#
# M^-1 = L (N^dag N)^-1 R + S
#
# R = N^dag ( 1   - EO OO^-1 ) ; R^dag = (1 - EO OO^-1)^dag N
#
#     ( EE^-1           ) 
# L = ( -OO^-1 OE EE^-1 )
#
#     ( 0   0     )
# S = ( 0   OO^-1 )
#
# All of the above also work if we interchange E<>O .  This therefore defines
# two preconditioners.  Depending on if N acts on even or odd sites,
# we call the corresponding version even/odd parity.
#

class eo2:
    def __init__(self, op, parity = None):
        self.op = op
        self.otype = op.otype
        self.parity = gpt.odd if parity is None else parity
        self.F_grid_eo = op.F_grid_eo
        self.F_grid = op.F_grid
        self.U_grid = op.U_grid
        self.tmp = gpt.lattice(self.F_grid_eo,self.otype)
        self.tmp2 = gpt.lattice(self.F_grid_eo,self.otype)
        self.ImportPhysicalFermionSource = self.op.ImportPhysicalFermionSource
        self.ExportPhysicalFermionSolution = self.op.ExportPhysicalFermionSolution
        self.Dminus = self.op.Dminus
        self.ExportPhysicalFermionSource = self.op.ExportPhysicalFermionSource

        def _N(op, ip):
            self.op.Mooee.inv_mat(self.tmp2,ip)
            self.op.Meooe.mat(op,self.tmp2)
            self.op.Mooee.inv_mat(self.tmp2,op)
            self.op.Meooe.mat(op,self.tmp2)
            op @= ip - op

        def _NDag(op, ip):
            self.op.Meooe.adj_mat(self.tmp2,ip)
            self.op.Mooee.adj_inv_mat(op,self.tmp2)
            self.op.Meooe.adj_mat(self.tmp2,op)
            self.op.Mooee.adj_inv_mat(op,self.tmp2)
            op @= ip - op

        def _NDagN(op, ip):
            _N(self.tmp,ip)
            _NDag(op,self.tmp)

        self.N = gpt.matrix_operator(mat = _N, adj_mat = _NDag, otype = op.otype, grid = self.F_grid_eo, 
                                     cb = self.parity)
        self.NDagN = gpt.matrix_operator(mat = _NDagN, adj_mat = _NDagN, otype = op.otype, grid = self.F_grid_eo,
                                         cb = self.parity)

    def import_parity(self,e,o):
        if self.parity is gpt.odd:
            return o,e
        return e,o

    def R(self, op, ie, io):
        ip,inp=self.import_parity(ie,io)
        self.op.Mooee.inv_mat(self.tmp,inp)
        self.op.Meooe.mat(op,self.tmp)
        self.tmp @= ip - op
        self.N.adj_mat(op,self.tmp)

    def RDag(self, oe, oo, ip):
        op,onp=self.import_parity(oe,oo)
        # R^dag = (1 - EO OO^-1)^dag N
        self.N.mat(onp,ip)
        self.op.Meooe.adj_mat(self.tmp,onp)
        self.op.Mooee.adj_inv_mat(op,self.tmp)
        op @= -op

    def L(self, oe, oo, ip):
        op,onp=self.import_parity(oe,oo)
        self.op.Mooee.inv_mat(op,ip)
        self.op.Meooe.mat(self.tmp,op)
        self.op.Mooee.inv_mat(onp,self.tmp)
        onp @= - onp

    def S(self, oe, oo, ie, io):
        ip,inp=self.import_parity(ie,io)
        op,onp=self.import_parity(oe,oo)
        self.op.Mooee.inv_mat(onp,inp)
        op[:]=0
        op.checkerboard(self.parity)
