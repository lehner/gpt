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
import cgpt

class operator:
    def __init__(self, name, U, params, Ls = None):
        self.name = name
        self.U = U
        self.U_grid = U[0].grid
        self.U_grid_eo = gpt.grid(self.U_grid.gdimensions,self.U_grid.precision,gpt.redblack)
        if Ls is None:
            self.F_grid = self.U_grid
            self.F_grid_eo = self.U_grid_eo
        else:
            self.F_grid = gpt.grid([ Ls ] + self.U_grid.gdimensions,self.U_grid.precision)
            self.F_grid_eo = gpt.grid(self.F_grid.gdimensions,self.U_grid.precision,gpt.redblack)

        self.params = {
            "U_grid" : self.U_grid.obj,
            "U_grid_rb" : self.U_grid_eo.obj,
            "F_grid" : self.F_grid.obj,
            "F_grid_rb" : self.F_grid_eo.obj,
            "U" : [ u.v_obj[0] for u in self.U ]
        }

        for k in params:
            assert(not k in [ "U_grid", "U_grid_rb", "F_grid", "F_grid_rb", "U" ])
            self.params[k] = params[k]

        self.obj = cgpt.create_fermion_operator(name,self.U_grid.precision,self.params)

        # register matrix operations
        gpt.qcd.fermion.register(self)

    def __del__(self):
        cgpt.delete_fermion_operator(self.obj)

    def unary(self, opcode, i, o):
        assert(len(i.v_obj) == 1)
        assert(len(o.v_obj) == 1)
        return cgpt.apply_fermion_operator(self.obj,opcode,i.v_obj[0],o.v_obj[0])

    def G5M(self, i, o):
        self.M(i,o)
        o @= gpt.gamma[5] * o

    
