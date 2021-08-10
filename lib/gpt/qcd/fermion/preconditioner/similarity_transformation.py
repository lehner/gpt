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


class similarity_transformation_instance:
    def __init__(self, base, transformation):

        tmp = gpt.lattice(base.S.grid[0], base.S.otype[0])

        def _R(op, i):
            transformation.mat(tmp, i)
            base.R.mat(op, tmp)

        def _L(o, ip):
            base.L.mat(tmp, ip)
            transformation.inv_mat(o, tmp)

        def _L_inv(op, i):
            transformation.mat(tmp, i)
            base.L.inv_mat(op, tmp)

        def _S(o, i):
            transformation.mat(o, i)
            base.S.mat(tmp, o)
            transformation.inv_mat(o, tmp)

        self.R = gpt.matrix_operator(
            mat=_R,
            otype=base.R.otype,
            grid=base.R.grid,
            cb=base.R.cb,
        )

        self.L = gpt.matrix_operator(
            mat=_L,
            inv_mat=_L_inv,
            otype=base.L.otype,
            grid=base.L.grid,
            cb=base.L.cb,
        )

        self.S = gpt.matrix_operator(
            mat=_S,
            otype=base.S.otype,
            grid=base.S.grid,
            cb=base.S.cb,
        )

        self.Mpc = base.Mpc
        self.F_grid = base.F_grid


def similarity_transformed_matrix_operator(op, transformation, tmp):
    def st(mat, transformation):
        def m(dst, src):
            transformation.inv_mat(dst, src)
            mat(tmp, dst)
            transformation.mat(dst, tmp)

        return m

    return gpt.matrix_operator(
        mat=st(op.mat, transformation),  # V mat V^-1
        inv_mat=st(op.inv_mat, transformation),  # V mat^-1 V^-1
        adj_mat=st(op.adj_mat, transformation.adj().inv()),  # V^-1^adj mat^adj V^adj
        adj_inv_mat=st(
            op.adj_inv_mat, transformation.adj().inv()
        ),  # V^-1^adj mat^adj^-1 V^adj
    )


class similarity_transformed_fermion_operator(gpt.matrix_operator):
    def __init__(self, base, transformation):

        super().__init__(mat=None, otype=base.otype, grid=base.grid, cb=base.cb)

        self.base = base
        self.transformation = transformation

        if hasattr(self.base, "F_grid_eo"):
            self.F_grid_eo = self.base.F_grid_eo
            tmp_eo = gpt.lattice(self.F_grid_eo, base.otype[0])
            self.Mooee = similarity_transformed_matrix_operator(
                self.base.Mooee, transformation, tmp_eo
            )
            self.Meooe = similarity_transformed_matrix_operator(
                self.base.Meooe, transformation, tmp_eo
            )

        self.F_grid = self.base.F_grid
        self.U_grid = self.base.U_grid
        self.ImportPhysicalFermionSource = self.base.ImportPhysicalFermionSource
        self.ExportPhysicalFermionSolution = self.base.ExportPhysicalFermionSolution
        self.ExportPhysicalFermionSource = self.base.ExportPhysicalFermionSource
        self.Dminus = self.base.Dminus


# D_V = V D V^-1 -> precond(D)_V = V^-1 precond(D_V) V
class similarity_transformation:
    def __init__(self, base, transformation):
        self.base = base
        self.transformation = transformation

    def __call__(self, op):
        self.base_instance = self.base(
            similarity_transformed_fermion_operator(op, self.transformation)
        )
        return similarity_transformation_instance(
            self.base_instance, self.transformation
        )
