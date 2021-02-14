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


class operator(gpt.matrix_operator):
    def __init__(self, name, U, params, otype, with_even_odd):
        # keep constructor parameters
        self.name = name
        self.U = U
        self.params_constructor = params

        # derived objects
        self.U_grid = U[0].grid
        if with_even_odd:
            self.U_grid_eo = gpt.grid(
                self.U_grid.gdimensions,
                self.U_grid.precision,
                gpt.redblack,
                parent=self.U_grid.parent,
                mpi=self.U_grid.mpi,
            )
        if "Ls" not in params:
            self.F_grid = self.U_grid
            if with_even_odd:
                self.F_grid_eo = self.U_grid_eo
        else:
            self.F_grid = self.U_grid.inserted_dimension(0, params["Ls"])
            if with_even_odd:
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
            "F_grid": self.F_grid.obj,
            "U": [u.v_obj[0] for u in self.U],
        }

        if with_even_odd:
            self.params["U_grid_rb"] = self.U_grid_eo.obj
            self.params["F_grid_rb"] = self.F_grid_eo.obj

        for k in params:
            assert k not in ["U_grid", "U_grid_rb", "F_grid", "F_grid_rb", "U"]
            self.params[k] = params[k]

        # register matrix operators
        class registry:
            pass

        gpt.qcd.fermion.register(registry, self)

        # map Grid matrix operations to clean matrix_operator structure
        super().__init__(
            mat=registry.M, adj_mat=registry.Mdag, otype=otype, grid=self.F_grid
        )

        if with_even_odd:
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
            self.DhopEO = gpt.matrix_operator(
                mat=registry.DhopEO,
                adj_mat=registry.DhopEODag,
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
        self.Dhop = gpt.matrix_operator(
            mat=registry.Dhop,
            adj_mat=registry.DhopDag,
            otype=otype,
            grid=self.F_grid,
        )
        self._Mdir = registry.Mdir

    def Mdir(self, mu, fb):
        return gpt.matrix_operator(
            mat=lambda dst, src: self._Mdir(dst, src, mu, fb),
            otype=self.otype,
            grid=self.F_grid,
        )

    @params_convention()
    def modified(self, params):
        return type(self)(
            name=self.name,
            U=self.U,
            params={**self.params_constructor, **params},
            otype=self.otype[0],
        )

    def converted(self, dst_precision):
        return self.updated(gpt.convert(self.U, dst_precision))

    @params_convention(make_hermitian=False)
    def coarsened(self, coarse_grid, basis, params):
        # TODO: allow for non-nearest-neighbor operators as well
        return gpt.qcd.fermion.coarse.nearest_neighbor_operator(
            self, coarse_grid, basis, params
        )

    def updated(self, U):
        return type(self)(
            name=self.name,
            U=U,
            params=self.params_constructor,
            otype=self.otype[0],
        )

    def update(self, U):
        self.U = U
        self.params["U"] = [u.v_obj[0] for u in U]
        cgpt.update_fermion_operator(self.obj, self.params)

    def split(self, mpi_split):
        split_grid = self.U_grid.split(mpi_split, self.U_grid.fdimensions)
        U_split = [gpt.lattice(split_grid, x.otype) for x in self.U]
        pos_split = gpt.coordinates(U_split[0])
        for i, x in enumerate(U_split):
            x[pos_split] = self.U[i][pos_split]
        return self.updated(U_split)

    def _G5M(self, dst, src):
        self(dst, src)
        dst @= gpt.qcd.fermion.coarse.gamma5(dst) * dst

    def propagator(self, solver):
        exp = self.ExportPhysicalFermionSolution
        imp = self.ImportPhysicalFermionSource

        inv_matrix = solver(self)

        def prop(dst_sc, src_sc):
            gpt.eval(dst_sc, exp * inv_matrix * imp * src_sc)

        return gpt.matrix_operator(
            prop,
            otype=(exp.otype[0], imp.otype[1]),
            grid=(exp.grid[0], imp.grid[1]),
            accept_list=True,
        )


class fine_operator(operator):
    def __init__(self, name, U, params, otype=None):
        super().__init__(name, U, params, otype, True)

        self.obj = cgpt.create_fermion_operator(
            name, self.U_grid.precision, self.params
        )

    def __del__(self):
        cgpt.delete_fermion_operator(self.obj)

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


class coarse_operator(operator):
    def __init__(self, name, U, params, otype=None):
        super().__init__(name, U, params, otype, True)
        self.tmp = gpt.lattice(self.F_grid, otype)
        self.tmp_eo = gpt.lattice(self.F_grid_eo, otype)
        self.U_self_inv = gpt.matrix.inv(self.U[8])

        self.obj = []
        for i in range(len(U[0].v_obj)):
            self.params["U"] = [a.v_obj[i] for a in self.U]
            self.params["U_self_inv"] = self.U_self_inv.v_obj[i]
            self.params["dag_factor"] = gpt.qcd.fermion.coarse.prefactor_dagger(
                self.U[8], i
            )
            self.obj.append(
                cgpt.create_fermion_operator(
                    self.name, self.U_grid.precision, self.params
                )
            )

    def __del__(self):
        for elem in self.obj:
            cgpt.delete_fermion_operator(elem)

    def apply_unary_operator(self, opcode, o, i):
        assert len(i.v_obj) == len(o.v_obj)
        assert len(i.v_obj) == (len(self.obj)) ** 0.5
        if len(i.v_obj) == 1:
            cgpt.apply_fermion_operator(self.obj[0], opcode, i.v_obj[0], o.v_obj[0])
        else:
            tmp = self.tmp if o.checkerboard() is gpt.none else self.tmp_eo
            for n in range(len(i.v_obj)):
                for m in range(len(i.v_obj)):
                    cgpt.apply_fermion_operator(
                        self.obj[n * len(i.v_obj) + m],
                        opcode,
                        i.v_obj[n],
                        o.v_obj[m] if n == 0 else tmp.v_obj[m],
                    )
                if n != 0:
                    o += tmp

    def apply_dirdisp_operator(self, opcode, o, i, direction, disp):
        assert len(i.v_obj) == len(o.v_obj)
        assert len(i.v_obj) == (len(self.obj)) ** 0.5
        if len(i.v_obj) == 1:
            cgpt.apply_fermion_operator_dirdisp(
                self.obj[0],
                opcode,
                i.v_obj[0],
                o.v_obj[0],
                direction,
                disp,
            )
        else:
            tmp = self.tmp  # dirdisp is on full grid by definition
            for n in range(len(i.v_obj)):
                for m in range(len(i.v_obj)):
                    cgpt.apply_fermion_operator_dirdisp(
                        self.obj[n * len(i.v_obj) + m],
                        opcode,
                        i.v_obj[n],
                        o.v_obj[m] if n == 0 else tmp.v_obj[m],
                        direction,
                        disp,
                    )
                if n != 0:
                    o += tmp
