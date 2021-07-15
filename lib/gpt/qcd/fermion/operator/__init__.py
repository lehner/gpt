#
#    GPT - Grid Python Toolkit
#    Copyright (C) 2020  Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)
#                  2020  Daniel Richtmann (daniel.richtmann@ur.de)
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
import gpt, cgpt
from gpt.params import params_convention
from gpt.qcd.fermion.operator.base import operator


class fine_operator(operator):
    def __init__(self, name, U, params, otype=None):
        super().__init__(name, U, params, otype, True)

        self.obj = cgpt.create_fermion_operator(
            name, self.U_grid.precision, self.params
        )

    def __del__(self):
        cgpt.delete_fermion_operator(self.obj)

    def apply_unary_operator(self, opcode, o, i):
        # Grid has different calling conventions which we adopt in cgpt:
        return cgpt.apply_fermion_operator(self.obj, opcode, i.v_obj, o.v_obj)

    def apply_dirdisp_operator(self, opcode, o, i, dir, disp):
        # Grid has different calling conventions which we adopt in cgpt:
        return cgpt.apply_fermion_operator_dirdisp(
            self.obj, opcode, i.v_obj, o.v_obj, dir, disp
        )

    def apply_deriv_operator(self, opcode, m, u, v):
        # Grid has different calling conventions which we adopt in cgpt:
        return cgpt.apply_fermion_operator_deriv(
            self.obj, opcode, [y for x in m for y in x.v_obj], u.v_obj, v.v_obj
        )


class differentiable_fine_operator(fine_operator):
    def __init__(self, name, U, params, otype=None):
        super().__init__(name, U, params, otype)

    # Future plan for interface:
    # M.projected_gradient(..)
    # M.adj().projected_gradient(...)
    # M.Meooe.projected_gradient(...)
    def deriv_core(self, func, left, right):
        _left = gpt.core.util.to_list(left)
        _right = gpt.core.util.to_list(right)
        assert len(_left) == len(_right)
        ders = []
        nd = len(self.U)
        ot = self.U[0].otype.cartesian()
        gg = self.U_grid
        ders = [gpt.lattice(gg, ot) for _ in range(nd * len(_left))]
        for i in range(len(_left)):
            func(ders[i * nd : (i + 1) * nd], _left[i], _right[i])

        # different convention in group generators
        # (-1j) * Ta^GRID = Ta^GPT
        # additional -1 due to Grid
        for d in ders:
            # d @= gpt.qcd.gauge.project.traceless_anti_hermitian(d)
            d @= (1j) * d
        return ders

    # TODO: Meoderiv,... they always return a full checkerboard force
    # which means some zeros are copied (and later added) unnecessarily
    # To change this behavior must edit cgpt/lib/operators/deriv.h
    def Mderiv(self, left, right):
        return self.deriv_core(self._MDeriv, left, right)

    def MderivDag(self, left, right):
        return self.deriv_core(self._MDerivDag, left, right)

    def Meoderiv(self, left, right):
        return self.deriv_core(self._MeoDeriv, left, right)

    def MeoderivDag(self, left, right):
        return self.deriv_core(self._MeoDerivDag, left, right)

    def Moederiv(self, left, right):
        return self.deriv_core(self._MoeDeriv, left, right)

    def MoederivDag(self, left, right):
        return self.deriv_core(self._MoeDerivDag, left, right)


class coarse_operator(operator):
    def __init__(self, name, U, params, otype=None):
        super().__init__(name, U, params, otype, True)
        self.tmp = gpt.lattice(self.F_grid, otype)
        self.tmp_eo = gpt.lattice(self.F_grid_eo, otype)
        self.U_self_inv = gpt.matrix.inv(self.U[8])
        self.t = gpt.timer("coarse_operator")

        self.params["U"] = [v_obj for u in U for v_obj in u.v_obj]
        self.params["U_self_inv"] = self.U_self_inv.v_obj
        self.obj = cgpt.create_fermion_operator(
            self.name, self.U_grid.precision, self.params
        )

    def __del__(self):
        cgpt.delete_fermion_operator(self.obj)

    def apply_unary_operator(self, opcode, o, i):
        cgpt.apply_fermion_operator(self.obj, opcode, i.v_obj, o.v_obj)

    def apply_dirdisp_operator(self, opcode, o, i, direction, disp):
        cgpt.apply_fermion_operator_dirdisp(
            self.obj,
            opcode,
            i.v_obj,
            o.v_obj,
            direction,
            disp,
        )
