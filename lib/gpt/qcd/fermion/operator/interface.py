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


class interface:
    def __init__(self):
        self.obj = None

    def setup(self, name, grid, params):
        self.obj = cgpt.create_fermion_operator(name, grid.precision, params)

    def __del__(self):
        if self.obj is not None:
            cgpt.delete_fermion_operator(self.obj)

    def update(self, params):
        cgpt.update_fermion_operator(self.obj, params)

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
