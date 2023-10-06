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

        if "omega" not in params:
            params["c_s"] = np.array([params["c"]] * params["Ls"], dtype=np.complex128)
        else:
            params["c_s"] = np.array(
                [
                    0.5 * (1.0 / omega * (params["b"] + params["c"]) - (params["b"] - params["c"]))
                    for omega in params["omega"]
                ],
                dtype=np.complex128,
            )
        params["b_s"] = np.array(
            [params["b"] - params["c"] + c for c in params["c_s"]], dtype=np.complex128
        )

        differentiable_fine_operator.__init__(self, name, U, params, otype, daggered)

        separate_cache = {}
        self.separate_cache = separate_cache

        def _J5q(dst4d, src5d):
            src4d = gpt.separate(src5d, 0, separate_cache)
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

        Dhop_projected_gradient = self.Dhop_projected_gradient
        ImportUnphysicalFermion = self.ImportUnphysicalFermion
        if daggered:
            ImportUnphysicalFermion = ImportUnphysicalFermion.adj()
            Dhop_projected_gradient = Dhop_projected_gradient.adj()

        c_s = params["c_s"]

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

        Ls = self.F_grid.fdimensions[0]
        rev0_cache = [{} for i in range(Ls)]

        def _rev0(dst, src):
            star = [slice(None, None, None)] * (src.grid.nd - 1)
            for i in range(Ls):
                dst[tuple([i] + star), rev0_cache[i]] = src[
                    tuple([Ls - 1 - i] + star), rev0_cache[i]
                ]

        self.R = gpt.matrix_operator(
            _rev0,
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

    def conserved_current(
        self, psi_left, src_left, psi_right, src_right, mu, sign, psi_left_flavor=None
    ):
        # sign = +1 (vector), -1 (axial vector)

        mass_plus = self.params["mass_plus"]
        mass_minus = self.params["mass_minus"]

        psi_right_shift = self.covariant_shift()

        assert not self.daggered
        assert mass_plus == mass_minus
        mass = mass_plus

        psi_left_shift = (
            psi_left_flavor.covariant_shift() if psi_left_flavor is not None else psi_right_shift
        )

        L_Q = gpt.separate(psi_left, 0, self.separate_cache)
        R_Q = gpt.separate(psi_right, 0, self.separate_cache)
        Ls = len(L_Q)

        Pplus = (gpt.gamma["I"].tensor() + gpt.gamma[5].tensor()) * 0.5
        Pminus = (gpt.gamma["I"].tensor() - gpt.gamma[5].tensor()) * 0.5

        L_Q_4d = gpt(Pminus * L_Q[0] + Pplus * L_Q[Ls - 1])
        R_Q_4d = gpt(Pminus * R_Q[0] + Pplus * R_Q[Ls - 1])

        L_TopRowWithSource = gpt(src_left + (1.0 - mass) * L_Q_4d)
        R_TopRowWithSource = gpt(src_right + (1.0 - mass) * R_Q_4d)

        TermA = gpt(Pplus * L_Q_4d)
        TermB = gpt(Pminus * L_Q_4d)
        TermC = gpt(Pminus * L_TopRowWithSource)
        TermD = gpt(Pplus * L_TopRowWithSource)

        L_TmLsGq0 = gpt(TermD - TermA + TermB)
        L_TmLsTmp = gpt(TermC - TermB + TermA)

        TermA = gpt(Pplus * R_Q_4d)
        TermB = gpt(Pminus * R_Q_4d)
        TermC = gpt(Pminus * R_TopRowWithSource)
        TermD = gpt(Pplus * R_TopRowWithSource)

        R_TmLsGq0 = gpt(TermD - TermA + TermB)
        R_TmLsTmp = gpt(TermC - TermB + TermA)

        R_TmLsGq = [gpt(Pminus * R_Q[s] + Pplus * R_Q[(s - 1 + Ls) % Ls]) for s in range(Ls)]
        L_TmLsGq = [gpt(Pminus * L_Q[s] + Pplus * L_Q[(s - 1 + Ls) % Ls]) for s in range(Ls)]

        dst = gpt.lattice(src_left)
        dst[:] = 0

        for s in range(Ls):
            sp = (s + 1) % Ls
            sr = Ls - 1 - s
            srp = (sr + 1) % Ls

            b = self.params["b_s"][s]
            c = self.params["c_s"][s]
            bpc = -0.5 / (b + c)

            if s == 0:
                p5d = gpt(
                    b * Pminus * L_TmLsGq[Ls - 1]
                    + c * Pplus * L_TmLsGq[Ls - 1]
                    + b * Pplus * L_TmLsTmp
                    + c * Pminus * L_TmLsTmp
                )
                tmp = gpt(
                    b * Pminus * R_TmLsGq0
                    + c * Pplus * R_TmLsGq0
                    + b * Pplus * R_TmLsGq[1]
                    + c * Pminus * R_TmLsGq[1]
                )
            elif s == Ls - 1:
                p5d = gpt(
                    b * Pminus * L_TmLsGq0
                    + c * Pplus * L_TmLsGq0
                    + b * Pplus * L_TmLsGq[1]
                    + c * Pminus * L_TmLsGq[1]
                )
                tmp = gpt(
                    b * Pminus * R_TmLsGq[Ls - 1]
                    + c * Pplus * R_TmLsGq[Ls - 1]
                    + b * Pplus * R_TmLsTmp
                    + c * Pminus * R_TmLsTmp
                )
            else:
                p5d = gpt(
                    b * Pminus * L_TmLsGq[sr]
                    + c * Pplus * L_TmLsGq[sr]
                    + b * Pplus * L_TmLsGq[srp]
                    + c * Pminus * L_TmLsGq[srp]
                )
                tmp = gpt(
                    b * Pminus * R_TmLsGq[s]
                    + c * Pplus * R_TmLsGq[s]
                    + b * Pplus * R_TmLsGq[sp]
                    + c * Pminus * R_TmLsGq[sp]
                )

            us_p5d = gpt(psi_right_shift.forward[mu] * tmp)
            gp5d = gpt(gpt.gamma[5] * p5d * gpt.gamma[5])
            gus_p5d = gpt(gpt.gamma[mu] * us_p5d)

            C = gpt(bpc * gpt.adj(gp5d) * (us_p5d - gus_p5d))

            if s == 0:
                p5d = gpt(
                    b * Pminus * R_TmLsGq0
                    + c * Pplus * R_TmLsGq0
                    + b * Pplus * R_TmLsGq[1]
                    + c * Pminus * R_TmLsGq[1]
                )
                tmp = gpt(
                    b * Pminus * L_TmLsGq[Ls - 1]
                    + c * Pplus * L_TmLsGq[Ls - 1]
                    + b * Pplus * L_TmLsTmp
                    + c * Pminus * L_TmLsTmp
                )
            elif s == Ls - 1:
                p5d = gpt(
                    b * Pminus * R_TmLsGq[Ls - 1]
                    + c * Pplus * R_TmLsGq[Ls - 1]
                    + b * Pplus * R_TmLsTmp
                    + c * Pminus * R_TmLsTmp
                )
                tmp = gpt(
                    b * Pminus * L_TmLsGq0
                    + c * Pplus * L_TmLsGq0
                    + b * Pplus * L_TmLsGq[1]
                    + c * Pminus * L_TmLsGq[1]
                )
            else:
                p5d = gpt(
                    b * Pminus * R_TmLsGq[s]
                    + c * Pplus * R_TmLsGq[s]
                    + b * Pplus * R_TmLsGq[sp]
                    + c * Pminus * R_TmLsGq[sp]
                )
                tmp = gpt(
                    b * Pminus * L_TmLsGq[sr]
                    + c * Pplus * L_TmLsGq[sr]
                    + b * Pplus * L_TmLsGq[srp]
                    + c * Pminus * L_TmLsGq[srp]
                )

            us_p5d = gpt(psi_left_shift.forward[mu] * tmp)

            gp5d = gpt(gpt.gamma[mu] * p5d)
            gus_p5d = gpt(gpt.gamma[5] * us_p5d * gpt.gamma[5])

            C -= gpt(bpc * gpt.adj(gus_p5d) * (gp5d + p5d))

            if s < Ls // 2:
                dst += sign * C
            else:
                dst += C

        return dst

    def conserved_vector_current(
        self, psi_left, src_left, psi_right, src_right, mu, psi_left_flavor=None
    ):
        return self.conserved_current(
            psi_left, src_left, psi_right, src_right, mu, +1.0, psi_left_flavor
        )

    def conserved_axial_current(
        self, psi_left, src_left, psi_right, src_right, mu, psi_left_flavor=None
    ):
        return self.conserved_current(
            psi_left, src_left, psi_right, src_right, mu, -1.0, psi_left_flavor
        )


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
    return mobius_class_operator("mobius", U, params, otype=gpt.ot_vector_spin_color(4, 3))
