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
import gpt, cgpt

class operator:
    def __init__(self, name, U, params, Ls = None, otype = None):

        # keep constructor parameters
        self.name = name
        self.U = U
        self.params_constructor = params
        self.Ls = Ls
        self.otype = otype

        # derived objects
        self.U_grid = U[0].grid
        self.U_grid_eo = gpt.grid(self.U_grid.gdimensions,self.U_grid.precision,gpt.redblack)
        if Ls is None:
            self.F_grid = self.U_grid
            self.F_grid_eo = self.U_grid_eo
        else:
            self.F_grid = gpt.grid([ Ls ] + self.U_grid.gdimensions,self.U_grid.precision)
            self.F_grid_eo = gpt.grid(self.F_grid.gdimensions,self.U_grid.precision,gpt.redblack)

        # parameter for create_fermion_operator
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

        # create fermion operator
        self.obj = cgpt.create_fermion_operator(name,self.U_grid.precision,self.params)

        # register matrix operators
        class registry:
            pass

        gpt.qcd.fermion.register(registry,self)

        # map Grid matrix operations to clean matrix_operator structure
        self.M = gpt.matrix_operator(mat = registry.M, adj_mat = registry.Mdag, otype = otype, grid = self.F_grid)
        self.Meooe = gpt.matrix_operator(mat = registry.Meooe, adj_mat = registry.MeooeDag, 
                                         otype = otype, grid = self.F_grid_eo)
        self.Mooee = gpt.matrix_operator(mat = registry.Mooee,
                                         adj_mat = registry.MooeeDag,
                                         inv_mat = registry.MooeeInv,
                                         adj_inv_mat = registry.MooeeInvDag,
                                         otype = otype, grid = self.F_grid_eo)
        self.Mdiag = gpt.matrix_operator(registry.Mdiag, otype = otype, grid = self.F_grid)
        self.Dminus = gpt.matrix_operator(mat = registry.Dminus, adj_mat = registry.DminusDag, otype = otype, grid = self.F_grid)
        self.ImportPhysicalFermionSource = gpt.matrix_operator(registry.ImportPhysicalFermionSource, 
                                                               otype = otype, grid = (self.F_grid,self.U_grid))
        self.ImportUnphysicalFermion = gpt.matrix_operator(registry.ImportUnphysicalFermion, 
                                                           otype = otype, grid = (self.F_grid,self.U_grid))
        self.ExportPhysicalFermionSolution = gpt.matrix_operator(registry.ExportPhysicalFermionSolution, 
                                                                 otype = otype, grid = (self.U_grid,self.F_grid))
        self.ExportPhysicalFermionSource = gpt.matrix_operator(registry.ExportPhysicalFermionSource, 
                                                               otype = otype, grid = (self.U_grid,self.F_grid))
        self.G5M = gpt.matrix_operator(lambda dst, src: self._G5M(dst,src), otype = otype, grid = self.F_grid)

    def __del__(self):
        cgpt.delete_fermion_operator(self.obj)

    def converted(self, dst_precision):
        if dst_precision == self.U[0].grid.precision:
            return self
        return operator(name = self.name,
                        U = gpt.convert(self.U, dst_precision),
                        params = self.params_constructor,
                        Ls = self.Ls,
                        otype = self.otype)

    def unary(self, opcode, o, i):
        assert(len(i.v_obj) == 1)
        assert(len(o.v_obj) == 1)
        # Grid has different calling conventions which we adopt in cgpt:
        return cgpt.apply_fermion_operator(self.obj,opcode,i.v_obj[0],o.v_obj[0])

    def _G5M(self, dst, src):
        self.M(dst,src)
        dst @= gpt.gamma[5] * dst

    
