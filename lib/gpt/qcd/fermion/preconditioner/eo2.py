#
# GPT
#
# Authors: Christoph Lehner 2020
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
# R = N^dag ( 1   - EO OO^-1 )
#
#     ( EE^-1           ) 
# L = ( -OO^-1 OE EE^-1 )
#
#     ( 0   0     )
# S = ( 0   OO^-1 )
#

class eo2:
    def __init__(self, op):
        self.op = op
        self.F_grid_eo = op.F_grid_eo
        self.F_grid = op.F_grid
        self.tmp = gpt.vspincolor(self.F_grid_eo)
        self.tmp2 = gpt.vspincolor(self.F_grid_eo)

    def import_physical(self, src, dst):
        self.op.ImportPhysicalFermionSource(src,dst)

    def export_physical(self, src, dst):
        self.op.ExportPhysicalFermionSolution(src,dst)

    def R(self, ie, io, oe):
        self.op.MooeeInv(io,self.tmp)
        self.op.Meooe(self.tmp,oe)
        self.tmp @= ie - oe
        self.NDag(self.tmp,oe)

    def L(self, ie, oe, oo):
        self.op.MooeeInv(ie,oe)
        self.op.Meooe(oe,self.tmp)
        self.op.MooeeInv(self.tmp,oo)
        oo @= - oo

    def S(self, ie, io, oe, oo):
        self.op.MooeeInv(io,oo)
        oe[:]=0
        oo @= io

    def NDagN(self, ie, oe):
        self.N(ie,self.tmp)
        self.NDag(self.tmp,oe)

    def N(self, ie, oe):
        self.op.MooeeInv(ie,self.tmp2)
        self.op.Meooe(self.tmp2,oe)
        self.op.MooeeInv(oe,self.tmp2)
        self.op.Meooe(self.tmp2,oe)
        oe @= ie - oe

    def NDag(self, ie, oe):
        self.op.MeooeDag(ie,self.tmp2)
        self.op.MooeeInvDag(self.tmp2,oe)
        self.op.MeooeDag(oe,self.tmp2)
        self.op.MooeeInvDag(self.tmp2,oe)
        oe @= ie - oe
