#
#    GPT - Grid Python Toolkit
#    Copyright (C) 2020  Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)
#                  2020  Daniel Richtmann (daniel.richtmann@ur.de)
#                  2022  Tristan Ueding
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
import gpt, copy
import numpy as np
from gpt.qcd.fermion.operator import differentiable_fine_operator


class mobius_class_operator(differentiable_fine_operator):
    def __init__(self, name, U, params, otype=None, daggered=False):

        if params["mass"] is not None:
            params["mass_plus"] = params["mass"]
            params["mass_minus"] = params["mass"]

        differentiable_fine_operator.__init__(self, name, U, params, otype, daggered)

        def _J5q(dst4d, src5d):
            src4d = gpt.separate(src5d, 0)
            Ls = len(src4d)
            # create correlator at the midpoint of the 5-th direction
            p_plus = gpt.eval(src4d[Ls // 2 - 1] + gpt.gamma[5] * src4d[Ls // 2 - 1])
            p_minus = gpt.eval(src4d[Ls // 2] - gpt.gamma[5] * src4d[Ls // 2])
            gpt.eval(dst4d, 0.5 * (p_plus + p_minus))

        self.J5q = gpt.matrix_operator(
            _J5q,
            vector_space=(self.vector_space_U, self.vector_space_F),
            accept_list=False,
        )

        self.bulk_propagator_to_propagator = self.ExportPhysicalFermionSolution

        # avoid reference loops
        if "omega" not in self.params:
            c_s = np.array([self.params["c"]] * self.params["Ls"], dtype=np.complex128)
        else:
            c_s = np.array(
                [
                    0.5
                    * (
                        1.0 / omega * (self.params["b"] + self.params["c"])
                        - (self.params["b"] - self.params["c"])
                    )
                    for omega in self.params["omega"]
                ],
                dtype=np.complex128,
            )
        Dhop_projected_gradient = self.Dhop_projected_gradient
        ImportUnphysicalFermion = self.ImportUnphysicalFermion
        if daggered:
            ImportUnphysicalFermion = ImportUnphysicalFermion.adj()
            Dhop_projected_gradient = Dhop_projected_gradient.adj()

        def _negative_surface_projection(src):
            src = gpt((-1.0) * ImportUnphysicalFermion * src)
            dst = gpt.lattice(src)
            gpt.scale_per_coordinate(dst, src, c_s, 0)
            return dst

        op = gpt.projected_matrix_operator(
            lambda left, right: Dhop_projected_gradient.mat(
                left, _negative_surface_projection(right)
            ),
            lambda left, right: Dhop_projected_gradient.adj_mat(
                _negative_surface_projection(left), right
            ),
            grid=(self.F_grid, self.U_grid),
            otype=(otype, otype),
            parity=gpt.full,
        )

        if daggered:
            op = op.adj()

        self.ImportPhysicalFermionSource_projected_gradient = op

        self.R = gpt.matrix_operator(
            lambda dst, src: gpt.eval(
                dst, gpt.merge(list(reversed(gpt.separate(gpt(src), 0))), 0)
            ),
            vector_space=self.vector_space,
            accept_list=False,
        )

    def bulk_propagator(self, solver):
        imp = self.ImportPhysicalFermionSource

        inv_matrix = solver(self)

        def prop(dst_sc, src_sc):
            gpt.eval(dst_sc, inv_matrix * imp * src_sc)

        op = gpt.matrix_operator(
            prop,
            vector_space=imp.vector_space,
            accept_list=True,
        )

        if self.daggered:
            op = op.adj()

        return op


@gpt.params_convention(
    mass=None,
    mass_plus=None,
    mass_minus=None,
    b=None,
    c=None,
    M5=None,
    boundary_phases=None,
    Ls=None,
)
def mobius(U, params):
    params = copy.deepcopy(params)  # save current parameters
    return mobius_class_operator(
        "mobius", U, params, otype=gpt.ot_vector_spin_color(4, 3)
    )
