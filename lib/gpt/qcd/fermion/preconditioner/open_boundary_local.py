#
#    GPT - Grid Python Toolkit
#    Copyright (C) 2022  Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)
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
import gpt as g
from gpt.algorithms import base_iterative


class open_boundary_local(base_iterative):
    @g.params_convention(margin=0)
    def __init__(self, inner_inverter, params):
        super().__init__()
        self.params = params
        self.margin = params["margin"]
        self.inner_inverter = inner_inverter

    def __call__(self, mat):
        # now create a local matrix
        U = mat.arguments()
        U_dim = len(U)
        U_grid = U[0].grid
        F_grid = mat.vector_space[1].grid
        F_dim = F_grid.nd

        U_ldomain = g.domain.local(U_grid, [self.margin] * U_dim)
        F_ldomain = g.domain.local(
            F_grid,
            [0] * (F_dim - U_dim) + [self.margin] * U_dim,
            cb=mat.vector_space[1].lattice().checkerboard(),
        )

        U_local = []
        colon = slice(None, None, None)

        for i in range(U_dim):
            u_local = U_ldomain.lattice(U[i].otype)
            U_ldomain.project(u_local, U[i])
            u_local[
                tuple([colon] * i + [u_local.grid.ldimensions[i] - 1] + [colon] * (U_dim - i - 1))
            ] = 0
            U_local.append(u_local)

        # get local matrix
        mat_local = mat.updated(U_local)

        F_ldomain_bulk = F_ldomain.bulk()
        F_ldomain_margin = F_ldomain.margin()

        # now need copy plans
        z = mat_local.vector_space[0].lattice()
        z[:] = 0

        mat_local_mat = mat_local.mat

        timer = [g.timer()]

        def proj_mat_local(dst, src):
            t = timer[0]
            t("local matrix")
            mat_local_mat(dst, src)
            t("zero projection")
            F_ldomain_margin.project(dst, z)
            t()

        # pml = g.matrix_operator(mat=proj_mat_local, vector_space=mat_local.vector_space).inherit(
        pml_inv = self.inner_inverter(proj_mat_local)

        # vector space
        vector_space = None
        if isinstance(mat, g.matrix_operator):
            vector_space = mat.vector_space

        _src = mat_local.vector_space[1].lattice()
        _dst = mat_local.vector_space[0].lattice()
        _src[:] = 0
        _dst[:] = 0

        def proj_mat(dst, src):
            F_ldomain_bulk.project(_src, src)
            proj_mat_local(_dst, _src)
            F_ldomain_bulk.promote(dst, _dst)

        @self.timed_function
        def inv(dst, src, t):
            timer[0] = t
            t("project")
            F_ldomain_bulk.project(_src, src)
            F_ldomain_bulk.project(_dst, dst)
            F_ldomain_margin.project(_dst, z)
            t("inner inverter")
            pml_inv(_dst, _src)
            t("promote")
            F_ldomain_bulk.promote(dst, _dst)
            t()

        return g.matrix_operator(
            mat=inv, inv_mat=proj_mat, accept_guess=(True, False), vector_space=vector_space
        )
