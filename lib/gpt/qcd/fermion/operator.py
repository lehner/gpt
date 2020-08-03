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
from gpt.params import params_convention


class operator(gpt.matrix_operator):
    def __init__(self, name, U, params, Ls=None, otype=None):

        # keep constructor parameters
        self.name = name
        self.U = U
        self.params_constructor = params
        self.Ls = Ls

        # derived objects
        self.U_grid = U[0].grid
        self.U_grid_eo = gpt.grid(
            self.U_grid.gdimensions,
            self.U_grid.precision,
            gpt.redblack,
            parent=self.U_grid.parent,
            mpi=self.U_grid.mpi,
        )
        if Ls is None:
            self.F_grid = self.U_grid
            self.F_grid_eo = self.U_grid_eo
        else:
            self.F_grid = self.U_grid.inserted_dimension(0, Ls)
            self.F_grid_eo = gpt.grid(
                self.F_grid.gdimensions,
                self.U_grid.precision,
                gpt.redblack,
                parent=self.F_grid.parent,
                mpi=self.F_grid.mpi,
            )

        # parameter for create_fermion_operator
        self.params = {
            "U_grid": self.U_grid.obj,
            "U_grid_rb": self.U_grid_eo.obj,
            "F_grid": self.F_grid.obj,
            "F_grid_rb": self.F_grid_eo.obj,
            "U": [u.v_obj[0] for u in self.U],
        }

        for k in params:
            assert k not in ["U_grid", "U_grid_rb", "F_grid", "F_grid_rb", "U"]
            self.params[k] = params[k]

        # create fermion operator
        self.obj = cgpt.create_fermion_operator(
            name, self.U_grid.precision, self.params
        )

        # register matrix operators
        class registry:
            pass

        class registry_dd:
            pass

        gpt.qcd.fermion.register(registry, self)
        gpt.qcd.fermion.register_dirdisp(registry_dd, self)

        # map Grid matrix operations to clean matrix_operator structure
        super().__init__(
            mat=registry.M, adj_mat=registry.Mdag, otype=otype, grid=self.F_grid
        )
        self.Meooe = gpt.matrix_operator(
            mat=registry.Meooe,
            adj_mat=registry.MeooeDag,
            otype=otype,
            grid=self.F_grid_eo,
        )
        self.Mooee = gpt.matrix_operator(
            mat=registry.Mooee,
            adj_mat=registry.MooeeDag,
            inv_mat=registry.MooeeInv,
            adj_inv_mat=registry.MooeeInvDag,
            otype=otype,
            grid=self.F_grid_eo,
        )
        self.Mdiag = gpt.matrix_operator(registry.Mdiag, otype=otype, grid=self.F_grid)
        self.Dminus = gpt.matrix_operator(
            mat=registry.Dminus,
            adj_mat=registry.DminusDag,
            otype=otype,
            grid=self.F_grid,
        )
        self.ImportPhysicalFermionSource = gpt.matrix_operator(
            registry.ImportPhysicalFermionSource,
            otype=otype,
            grid=(self.F_grid, self.U_grid),
        )
        self.ImportUnphysicalFermion = gpt.matrix_operator(
            registry.ImportUnphysicalFermion,
            otype=otype,
            grid=(self.F_grid, self.U_grid),
        )
        self.ExportPhysicalFermionSolution = gpt.matrix_operator(
            registry.ExportPhysicalFermionSolution,
            otype=otype,
            grid=(self.U_grid, self.F_grid),
        )
        self.ExportPhysicalFermionSource = gpt.matrix_operator(
            registry.ExportPhysicalFermionSource,
            otype=otype,
            grid=(self.U_grid, self.F_grid),
        )
        self.G5M = gpt.matrix_operator(
            lambda dst, src: self._G5M(dst, src), otype=otype, grid=self.F_grid
        )
        self.Mdir = gpt.matrix_operator(
            mat=registry_dd.Mdir, otype=otype, grid=self.F_grid
        )

    def __del__(self):
        cgpt.delete_fermion_operator(self.obj)

    def updated(self, U):
        return operator(
            name=self.name,
            U=U,
            params=self.params_constructor,
            Ls=self.Ls,
            otype=self.otype[0],
        )

    def converted(self, dst_precision):
        return self.updated(gpt.convert(self.U, dst_precision))

    @params_convention()
    def modified(self, params):
        return operator(
            name=self.name,
            U=self.U,
            params={**self.params_constructor, **params},
            Ls=self.Ls,
            otype=self.otype[0],
        )

    def apply_unary_operator(self, opcode, o, i):
        assert len(i.v_obj) == 1
        assert len(o.v_obj) == 1
        # Grid has different calling conventions which we adopt in cgpt:
        return cgpt.apply_fermion_operator(self.obj, opcode, i.v_obj[0], o.v_obj[0])

    def apply_dirdisp_operator(self, opcode, o, i, dir, disp):
        assert len(i.v_obj) == 1
        assert len(o.v_obj) == 1
        # Grid has different calling conventions which we adopt in cgpt:
        return cgpt.apply_fermion_operator_dirdisp(
            self.obj, opcode, i.v_obj[0], o.v_obj[0], dir, disp
        )

    def _G5M(self, dst, src):
        self(dst, src)
        dst @= gpt.gamma[5] * dst

    def propagator(self, solver):
        exp = self.ExportPhysicalFermionSolution
        imp = self.ImportPhysicalFermionSource

        inv_matrix = solver(self)

        def prop(dst_sc, src_sc):
            inv_matrix(dst_sc, gpt.eval(imp * src_sc))
            dst_sc @= exp * dst_sc

        return gpt.matrix_operator(
            prop, otype=(exp.otype[0], imp.otype[1]), grid=(exp.grid[0], imp.grid[1])
        )


class coarse_operator(gpt.matrix_operator):
    def __init__(self, A, params, Ls=None, otype=None):

        # keep constructor parameters
        self.name = "coarse"
        self.A = A
        self.params_constructor = params
        self.Ls = Ls
        self.otype = otype

        # derived objects
        self.A_grid = A[0].grid
        # self.A_grid_eo = gpt.grid(
        #     self.A_grid.gdimensions, self.A_grid.precision, gpt.redblack
        # )
        if Ls is not None:
            assert Ls is None
        self.F_grid = self.A_grid
        # self.F_grid_eo = self.A_grid_eo
        # NOTE: The eo grids are not used currently

        # parameter for create_fermion_operator
        self.params = {
            "grid_c": self.A_grid.obj,
            "hermitian": False,  # NOTE: doesn't matter what we pass to Grid since we need to do Mdag ourselves anyway
            "level": params["level"],
            "nbasis": int(
                self.A[0].otype.shape[0] / (len(A[0].v_obj)) ** 0.5
            ),  # for one instance
            # "A": [a.v_obj[0] for a in self.A],
        }

        for k in params:
            self.params[k] = params[k]

        self.obj = []
        for i in range(len(A[0].v_obj)):
            self.params["A"] = [a.v_obj[i] for a in self.A]
            self.obj.append(
                cgpt.create_fermion_operator(
                    self.name, self.A_grid.precision, self.params
                )
            )

        # register matrix operators
        class registry:
            pass

        class registry_dd:
            pass

        gpt.qcd.fermion.register(registry, self)
        gpt.qcd.fermion.register_dirdisp(registry_dd, self)

        # map Grid matrix operations to clean matrix_operator structure
        super().__init__(
            mat=registry.M, adj_mat=registry.Mdag, otype=otype, grid=self.F_grid
        )
        self.ImportPhysicalFermionSource = gpt.matrix_operator(
            registry.ImportPhysicalFermionSource,
            otype=otype,
            grid=(self.F_grid, self.A_grid),
        )
        self.ExportPhysicalFermionSolution = gpt.matrix_operator(
            registry.ExportPhysicalFermionSolution,
            otype=otype,
            grid=(self.A_grid, self.F_grid),
        )
        self.ExportPhysicalFermionSource = gpt.matrix_operator(
            registry.ExportPhysicalFermionSource,
            otype=otype,
            grid=(self.A_grid, self.F_grid),
        )
        self.G5M = gpt.matrix_operator(
            lambda dst, src: self._G5M(dst, src), otype=otype, grid=self.F_grid
        )
        self.Mdir = gpt.matrix_operator(
            mat=registry_dd.Mdir, otype=otype, grid=self.F_grid
        )

    def __del__(self):
        for elem in self.obj:
            cgpt.delete_fermion_operator(elem)

    def updated(self, A):
        return coarse_operator(
            name=self.name,
            A=A,
            params=self.params_constructor,
            Ls=self.Ls,
            otype=self.otype[0],
        )

    def converted(self, dst_precision):
        return self.updated(gpt.convert(self.A, dst_precision))

    @params_convention()
    def modified(self, params):
        return operator(
            name=self.name,
            A=self.A,
            params={**self.params_constructor, **params},
            Ls=self.Ls,
            otype=self.otype[0],
        )

    def apply_unary_operator(self, opcode, o, i):
        assert len(i.v_obj) == len(o.v_obj)
        assert len(i.v_obj) == (len(self.obj)) ** 0.5
        tmp = gpt.lattice(o)
        o[:] = 0.0
        # Grid has different calling conventions which we adopt in cgpt:
        for m in range(len(i.v_obj)):
            tmp[:] = 0.0
            for n in range(len(i.v_obj)):
                cgpt.apply_fermion_operator(
                    self.obj[n * len(i.v_obj) + m], opcode, i.v_obj[n], tmp.v_obj[m]
                )
                o += tmp

    def apply_dirdisp_operator(self, opcode, o, i, direction, disp):
        assert len(i.v_obj) == len(o.v_obj)
        assert len(i.v_obj) == (len(self.obj)) ** 0.5
        tmp = gpt.lattice(o)
        o[:] = 0.0
        # Grid has different calling conventions which we adopt in cgpt:
        for m in range(len(i.v_obj)):
            tmp[:] = 0.0
            for n in range(len(i.v_obj)):
                cgpt.apply_fermion_operator_dirdisp(
                    self.obj[n * len(i.v_obj) + m],
                    opcode,
                    i.v_obj[n],
                    tmp.v_obj[m],
                    direction,
                    disp,
                )
                o += tmp

    def _G5M(self, dst, src):
        self(dst, src)
        g5 = gpt.g5c(dst)
        dst @= g5 * dst

    def propagator(self, solver):
        exp = self.ExportPhysicalFermionSolution
        imp = self.ImportPhysicalFermionSource

        inv_matrix = solver(self)

        def prop(dst_sc, src_sc):
            inv_matrix(dst_sc, gpt.eval(imp * src_sc))
            dst_sc @= exp * dst_sc

        return gpt.matrix_operator(
            prop, otype=(exp.otype[0], imp.otype[1]), grid=(exp.grid[0], imp.grid[1])
        )
