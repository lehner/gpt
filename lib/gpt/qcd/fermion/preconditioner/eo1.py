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
from gpt.params import params_convention

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
# M^-1 = L (N^dag N)^-1 R + S    for eo2_ne
# M^-1 = L N^-1 R + S            for eo2
# M^-1 = L Mpc^-1 R + S          general form
#
# R = N^dag EE^-1 ( 1   - EO OO^-1 )  ;  R^dag = ( 1   - EO OO^-1 )^dag EE^-1^dag N
#
#     ( 1         )
# L = ( -OO^-1 OE )
#
#     ( 0   0     )
# S = ( 0   OO^-1 )
#
# A2A:
#
# M^-1 = L |n><n| R + S = v w^dag + S ;  -> v = L |n>, w = R^dag |n>
#
# All of the above also work if we interchange E<>O .  This therefore defines
# two preconditioners.  Depending on if N acts on even or odd sites,
# we call the corresponding version even/odd parity.
#


class eo1_base:
    def __init__(self, op, parity):
        self.op = op
        self.otype = op.otype[0]
        self.parity = gpt.odd if parity is None else parity
        self.F_grid_eo = op.F_grid_eo
        self.F_grid = op.F_grid
        self.U_grid = op.U_grid
        self.tmp = gpt.lattice(self.F_grid_eo, self.otype)
        self.tmp2 = gpt.lattice(self.F_grid_eo, self.otype)  # need for nested call in R
        self.in_p = gpt.lattice(self.F_grid_eo, self.otype)
        self.in_np = gpt.lattice(self.F_grid_eo, self.otype)
        self.out_p = gpt.lattice(self.F_grid_eo, self.otype)
        self.out_np = gpt.lattice(self.F_grid_eo, self.otype)
        self.ImportPhysicalFermionSource = self.op.ImportPhysicalFermionSource
        self.ExportPhysicalFermionSolution = self.op.ExportPhysicalFermionSolution
        self.Dminus = self.op.Dminus
        self.ExportPhysicalFermionSource = self.op.ExportPhysicalFermionSource

        def _N(op, ip):
            self.op.Meooe.mat(self.tmp2, ip)
            self.op.Mooee.inv_mat(op, self.tmp2)
            self.op.Meooe.mat(self.tmp2, op)
            self.op.Mooee.inv_mat(op, self.tmp2)
            op @= ip - op

        def _NDag(op, ip):
            self.op.Mooee.adj_inv_mat(self.tmp2, ip)
            self.op.Meooe.adj_mat(op, self.tmp2)
            self.op.Mooee.adj_inv_mat(self.tmp2, op)
            self.op.Meooe.adj_mat(op, self.tmp2)
            op @= ip - op

        def _NDagN(op, ip):
            _N(self.tmp, ip)
            _NDag(op, self.tmp)

        def _L(o, ip):
            self.out_p @= ip
            self.op.Meooe.mat(self.tmp, ip)
            self.op.Mooee.inv_mat(self.out_np, self.tmp)
            self.out_np @= -self.out_np
            self.export_parity(o)

        def _L_pseudo_inverse(op, i):
            self.import_parity(i)
            op @= self.in_p

        def _S(o, i):
            self.import_parity(i)
            self.op.Mooee.inv_mat(self.out_np, self.in_np)
            self.out_p[:] = 0
            self.out_p.checkerboard(self.parity)
            self.export_parity(o)

        self.L = gpt.matrix_operator(
            mat=_L,
            inv_mat=_L_pseudo_inverse,
            otype=op.otype,
            grid=(self.F_grid, self.F_grid_eo),
            cb=(None, self.parity),
        )

        self.S = gpt.matrix_operator(
            mat=_S,
            otype=op.otype,
            grid=self.F_grid,
        )

        self.N = gpt.matrix_operator(
            mat=_N, adj_mat=_NDag, otype=op.otype, grid=self.F_grid_eo, cb=self.parity
        )

        self.NDagN = gpt.matrix_operator(
            mat=_NDagN,
            adj_mat=_NDagN,
            otype=op.otype,
            grid=self.F_grid_eo,
            cb=self.parity,
        )

        for undressed in ["N", "NDagN"]:
            self.__dict__[undressed].split = lambda mpi: eo1_base(
                op.split(mpi), parity
            ).__dict__[undressed]

    def import_parity(self, i):
        gpt.pick_checkerboard(self.parity, self.in_p, i)
        gpt.pick_checkerboard(self.parity.inv(), self.in_np, i)

    def export_parity(self, o):
        gpt.set_checkerboard(o, self.out_p)
        gpt.set_checkerboard(o, self.out_np)


class eo1_ne_instance(eo1_base):
    def __init__(self, op, parity):
        super().__init__(op, parity)

        def _R(op, i):
            self.import_parity(i)
            self.op.Mooee.inv_mat(self.tmp, self.in_np)
            self.op.Meooe.mat(op, self.tmp)
            op @= self.in_p - op
            self.op.Mooee.inv_mat(self.tmp, op)
            self.N.adj_mat(op, self.tmp)

        def _RDag(o, ip):
            # R^dag = ( 1   - EO OO^-1 )^dag EE^-1^dag N
            self.N.mat(self.out_np, ip)
            self.op.Mooee.adj_inv_mat(self.out_p, self.out_np)
            self.op.Meooe.adj_mat(self.tmp, self.out_p)
            self.op.Mooee.adj_inv_mat(self.out_np, self.tmp)
            self.out_np @= -self.out_np
            self.export_parity(o)

        self.R = gpt.matrix_operator(
            mat=_R,
            adj_mat=_RDag,
            otype=op.otype,
            grid=(self.F_grid_eo, self.F_grid),
            cb=(self.parity, None),
        )

        self.Mpc = self.NDagN


class eo1_ne:
    @params_convention(parity=None)
    def __init__(self, params):
        self.params = params

    def __call__(self, op):
        return eo1_ne_instance(op, self.params["parity"])


class eo1_instance(eo1_base):
    def __init__(self, op, parity):
        super().__init__(op, parity)

        def _R(self, op, i):
            self.import_parity(i)
            self.op.Mooee.inv_mat(self.tmp, self.in_np)
            self.op.Meooe.mat(op, self.tmp)
            self.tmp @= self.in_p - op
            self.op.Mooee.inv_mat(op, self.tmp)

        self.R = gpt.matrix_operator(
            mat=_R,
            otype=op.otype,
            grid=(self.F_grid_eo, self.F_grid),
            cb=(self.parity, None),
        )

        self.Mpc = self.N


class eo1:
    @params_convention(parity=None)
    def __init__(self, params):
        self.params = params

    def __call__(self, op):
        return eo1_instance(op, self.params["parity"])
