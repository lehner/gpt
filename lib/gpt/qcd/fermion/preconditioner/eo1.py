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
import gpt, sys

# First EO preconditioning (upper triangular left)
#
#      ( EE EO )   ( EE   EO OO^-1 ) ( N 0 ) ( 1  0  )
#  M = ( OE OO ) = ( 0    1        ) ( 0 1 ) ( OE OO )
#
#  N = 1 - EE^-1 EO OO^-1 OE
#
#  Verify:  ( EE  0  )   ( EE  0 ) ( 1   0  )         ( 1   EO OO^-1 )   ( EE   EO OO^-1 ) ( EE^-1   0 )
#           ( OE  OO ) = ( 0   1 ) ( OE  OO )   and   ( 0   1        ) = ( 0    1        ) ( 0       1 )   then combine with eo2.py
#
# Then:
#
#        ( 1  0  )^-1 ( N^dag^-1 (N^dag N)  0 )^-1 ( EE        EO OO^-1  )^-1
# M^-1 = ( OE OO )    ( 0                   1 )    ( 0         1         )
#
#        ( 1              0   )  ( (N^dag N)^-1 N^dag   0 )  ( EE^-1   - EE^-1 EO OO^-1 )
#      = ( -OO^-1 OE    OO^-1 )  ( 0                    1 )  ( 0              1         )
#
# M^-1 = L (N^dag N)^-1 R + S
#
# R = N^dag EE^-1 ( 1   - EO OO^-1 )
#
#     ( 1         ) 
# L = ( -OO^-1 OE )
#
#     ( 0   0     )
# S = ( 0   OO^-1 )
#

class eo1:
    def __init__(self, op):
        self.op = op
        self.F_grid_eo = op.F_grid_eo
        self.F_grid = op.F_grid
        self.tmp = gpt.vspincolor(self.F_grid_eo)
        self.tmp2 = gpt.vspincolor(self.F_grid_eo) # need for nested call in R

    def ImportPhysicalFermionSource(self, src, dst):
        self.op.ImportPhysicalFermionSource(src, dst)

    def ExportPhysicalFermionSolution(self, src, dst):
        self.op.ExportPhysicalFermionSolution(src, dst)

    def R(self, ie, io, oe):
        self.op.MooeeInv(io,self.tmp)
        self.op.Meooe(self.tmp,oe)
        oe @= ie - oe
        self.op.MooeeInv(oe,self.tmp)
        self.NDag(self.tmp,oe)

    def L(self, ie, oe, oo):
        oe @= ie
        self.op.Meooe(ie,self.tmp)
        self.op.MooeeInv(self.tmp,oo)
        oo @= - oo

    def S(self, ie, io, oe, oo):
        self.op.MooeeInv(io,oo)
        oe[:]=0

    def NDagN(self, ie, oe):
        self.N(ie,self.tmp)
        self.NDag(self.tmp,oe)

    def N(self, ie, oe):
        self.op.Meooe(ie,self.tmp2)
        self.op.MooeeInv(self.tmp2,oe)
        self.op.Meooe(oe,self.tmp2)
        self.op.MooeeInv(self.tmp2,oe)
        oe @= ie - oe

    def NDag(self, ie, oe):
        self.op.MooeeInvDag(ie,self.tmp2)
        self.op.MeooeDag(self.tmp2,oe)
        self.op.MooeeInvDag(oe,self.tmp2)
        self.op.MeooeDag(self.tmp2,oe)
        oe @= ie - oe
