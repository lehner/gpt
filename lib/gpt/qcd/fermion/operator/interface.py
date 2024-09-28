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
import cgpt
import gpt as g

operator_tag = {}
operator_limbo = {}

verbose = g.default.is_verbose("fermion-operator")


class interface:
    def __init__(self):
        self.obj = None

    def _setup(self, name, grid, params):
        assert self.obj is None

        tag_params = {
            x: params[x] for x in params if x not in ["U", "mass", "mass_plus", "mass_minus"]
        }
        tag = f"{name}_{grid.precision.cgpt_dtype}_{tag_params}"

        if tag in operator_limbo and len(operator_limbo[tag]) > 0:
            self.obj = operator_limbo[tag].pop()
            # set mass needs to precede update for clover-type fermions
            cgpt.set_mass_fermion_operator(self.obj, params)
            cgpt.update_fermion_operator(self.obj, params)

            if verbose:
                g.message(f"Re-used fermion operator {tag}")
        else:
            # create new operator
            self.obj = cgpt.create_fermion_operator(name, grid.precision.cgpt_dtype, params)
            operator_tag[self.obj] = tag

            if verbose:
                g.message("Status of allocated fermion operators:")
                statistics = {}
                for tag in operator_tag:
                    if operator_tag[tag] not in statistics:
                        statistics[operator_tag[tag]] = 1
                    else:
                        statistics[operator_tag[tag]] += 1
                for tag in statistics:
                    g.message(f" {statistics[tag]} of type {tag}")

    def setup(self, name, grid, params):
        self.setup_arguments = (name, grid, params)
        self._setup(*self.setup_arguments)

    def __del__(self):
        self.suspend()

    def suspend(self):
        if self.obj is not None:
            tag = operator_tag[self.obj]
            if tag not in operator_limbo:
                operator_limbo[tag] = [self.obj]
            else:
                operator_limbo[tag].append(self.obj)

            self.obj = None

    def update(self, params):
        if self.obj is None:
            # wake from suspended state
            self._setup(*self.setup_arguments)
        cgpt.update_fermion_operator(self.obj, params)

    def apply_unary_operator(self, opcode, o, i):
        assert self.obj is not None
        # Grid has different calling conventions which we adopt in cgpt:
        return cgpt.apply_fermion_operator(self.obj, opcode, i.v_obj, o.v_obj)

    def apply_dirdisp_operator(self, opcode, o, i, dir, disp):
        assert self.obj is not None
        # Grid has different calling conventions which we adopt in cgpt:
        return cgpt.apply_fermion_operator_dirdisp(self.obj, opcode, i.v_obj, o.v_obj, dir, disp)

    def apply_deriv_operator(self, opcode, m, u, v):
        assert self.obj is not None
        # Grid has different calling conventions which we adopt in cgpt:
        return cgpt.apply_fermion_operator_deriv(
            self.obj, opcode, [y for x in m for y in x.v_obj], u.v_obj, v.v_obj
        )
