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
import gpt
from gpt.params import params_convention


class method_registry:
    pass


class base(gpt.matrix_operator):
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
        registry = method_registry()
        gpt.qcd.fermion.register(registry, self.interface)

        # map Grid matrix operations to clean matrix_operator structure
        super().__init__(
            mat=registry.M, adj_mat=registry.Mdag, otype=otype, grid=self.F_grid
        )

        # create operator domain
        self.domain = gpt.core.domain.full(self.F_grid)

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
            self._MoeDeriv = registry.MoeDeriv
            self._MoeDerivDag = registry.MoeDerivDag
            self._MeoDeriv = registry.MeoDeriv
            self._MeoDerivDag = registry.MeoDerivDag

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

        def _G5M(dst, src):
            registry.M(dst, src)
            dst @= gpt.qcd.fermion.coarse.gamma5(dst) * dst

        self.G5M = gpt.matrix_operator(_G5M, otype=otype, grid=self.F_grid)
        self.Dhop = gpt.matrix_operator(
            mat=registry.Dhop,
            adj_mat=registry.DhopDag,
            otype=otype,
            grid=self.F_grid,
        )
        self._Mdir = registry.Mdir
        self._MDeriv = registry.MDeriv
        self._MDerivDag = registry.MDerivDag

    def Mdir(self, mu, fb):
        return gpt.matrix_operator(
            mat=lambda dst, src: self._Mdir(dst, src, mu, fb),
            otype=self.otype,
            grid=self.F_grid,
        )

    def modified(self, **params):
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
        self.interface.update(self.params)

    def split(self, mpi_split):
        split_grid = self.U_grid.split(mpi_split, self.U_grid.fdimensions)
        U_split = [gpt.lattice(split_grid, x.otype) for x in self.U]
        pos_split = gpt.coordinates(U_split[0])
        for i, x in enumerate(U_split):
            x[pos_split] = self.U[i][pos_split]
        return self.updated(U_split)

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

    def even_odd_sites_decomposed(self, parity):
        class even_odd_sites:
            def __init__(me):
                me.D_domain = gpt.domain.even_odd_sites(self.F_grid_eo, parity)
                me.C_domain = gpt.domain.even_odd_sites(self.F_grid_eo, parity.inv())
                me.DD = self.Mooee
                me.CC = self.Mooee
                me.CD = self.Meooe
                me.DC = self.Meooe

        return even_odd_sites()
