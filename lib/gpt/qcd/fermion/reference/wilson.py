#
#    GPT - Grid Python Toolkit
#    Copyright (C) 2020  Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)
#    Copyright (C) 2020  Daniel Richtmann (daniel.richtmann@ur.de, https://github.com/lehner/gpt)
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
from gpt.params import params_convention
from gpt.core.covariant import shift, apply_boundaries
from gpt import matrix_operator, site_diagonal_operator


class wilson(shift, matrix_operator):
    # M = sum_mu gamma[mu]*D[mu] + m0 - 1/2 sum_mu D^2[mu]
    # m0 + 4 = 1/2/kappa
    @params_convention()
    def __init__(self, U, params):

        shift.__init__(self, U, params)

        Nc = U[0].otype.Nc
        self.otype = g.ot_vector_spin_color(4, Nc)
        self.U_grid = U[0].grid
        self.U_grid_eo = g.grid(
            self.U_grid.gdimensions,
            self.U_grid.precision,
            g.redblack,
            parent=self.U_grid.parent,
            mpi=self.U_grid.mpi,
        )
        self.F_grid = self.U_grid
        self.F_grid_eo = self.U_grid_eo

        self.csw = None
        for one, other in [("csw_r", "csw_t"), ("csw_t", "csw_r")]:
            if one in params:
                assert other in params
                self.csw = params[one]
                assert params[other] == self.csw
        assert "csw" not in params

        if "mass" in params:
            assert "kappa" not in params
            self.mass = params["mass"]
            self.kappa = 1.0 / (params["mass"] + 4.0) / 2.0
        else:
            assert "kappa" in params
            self.kappa = params["kappa"]
            self.mass = 1 / (2 * params["kappa"]) - 4

        self.open_bc = params["boundary_phases"][3] == 0.0
        if self.open_bc:
            assert "cF" in params
            self.cF = params["cF"]
            T = self.L[3]

        if self.csw is not None:

            def gamma_product(M, gamma):
                """
                Ideally, this would just be 'g.gamma[gamma] * M',
                but GPTs expressions cant quite to that (yet?)
                """
                gamma = g.gamma[gamma].tensor()
                r = g.mspincolor(M.grid)
                r[:] = 0
                for mu in range(4):
                    for nu in range(4):
                        if gamma[mu, nu] != 0.0:
                            r[:, :, :, :, mu, nu, :, :] = gamma[mu, nu] * M[:, :, :, :, :, :]
                return r

            field_strength = g.qcd.gauge.field_strength
            self.clover = gamma_product(field_strength(self.U, 0, 1), "SigmaXY")
            self.clover += gamma_product(field_strength(self.U, 0, 2), "SigmaXZ")
            self.clover += gamma_product(field_strength(self.U, 0, 3), "SigmaXT")
            self.clover += gamma_product(field_strength(self.U, 1, 2), "SigmaYZ")
            self.clover += gamma_product(field_strength(self.U, 1, 3), "SigmaYT")
            self.clover += gamma_product(field_strength(self.U, 2, 3), "SigmaZT")

            if self.open_bc:  # unity at the temporal boundaries
                self.clover[:, :, :, 0, :, :, :, :] = 0.0
                self.clover[:, :, :, T - 1, :, :, :, :] = 0.0
                for alpha in range(4):
                    for a in range(Nc):
                        self.clover[:, :, :, 0, alpha, alpha, a, a] = 1.0
                        self.clover[:, :, :, T - 1, alpha, alpha, a, a] = 1.0

            self.clover *= -0.5 * self.csw
            for alpha in range(4):
                for a in range(Nc):
                    self.clover[:, :, :, :, alpha, alpha, a, a] += 0.5 / self.kappa

            # improvement coefficients next to temporal boundaries
            if self.open_bc and self.cF != 1.0:
                for alpha in range(4):
                    for a in range(Nc):
                        self.clover[:, :, :, 1, alpha, alpha, a, a] += self.cF - 1.0
                        self.clover[:, :, :, T - 2, alpha, alpha, a, a] += self.cF - 1.0

            self.Mdiag = site_diagonal_operator(self.clover)

        else:
            self.clover = 0.5 / self.kappa
            self.clover_inv = 1.0 / self.clover

            self.Mdiag = g.matrix_operator(
                mat=lambda dst, src: self._Mooee(dst, src),
                inv_mat=lambda dst, src: self._MooeeInv(dst, src),
                otype=self.otype,
                grid=self.F_grid,
            )

        self.Meooe = g.matrix_operator(
            lambda dst, src: self._Meooe(dst, src),
            otype=self.otype,
            grid=self.F_grid_eo,
        )
        self.Mooee = g.matrix_operator(
            lambda dst, src: self._Mooee(dst, src),
            otype=self.otype,
            grid=self.F_grid_eo,
        )
        matrix_operator.__init__(
            self, lambda dst, src: self._M(dst, src), otype=self.otype, grid=self.F_grid
        )
        self.Dhop = g.matrix_operator(
            lambda dst, src: self._Meooe(dst, src), otype=self.otype, grid=self.F_grid
        )
        self.M = g.matrix_operator(
            lambda dst, src: self._M(dst, src), otype=self.otype, grid=self.F_grid
        )
        self.G5M = g.matrix_operator(
            lambda dst, src: self._G5M(dst, src), otype=self.otype, grid=self.F_grid
        )
        self.ImportPhysicalFermionSource = g.matrix_operator(
            lambda dst, src: g.copy(dst, src),
            otype=self.otype,
            grid=(self.U_grid, self.F_grid),
        )
        self.ExportPhysicalFermionSolution = g.matrix_operator(
            lambda dst, src: g.copy(dst, src),
            otype=self.otype,
            grid=(self.U_grid, self.F_grid),
        )

    def Mdir(self, mu, fb):
        return g.matrix_operator(
            mat=lambda dst, src: self._Mdir(dst, src, mu, fb),
            otype=self.otype,
            grid=self.F_grid,
        )

    def _Meooe(self, dst, src):
        assert dst != src
        dst[:] = 0
        for mu in range(4):
            src_plus = g.eval(self.forward[mu] * src)
            src_minus = g.eval(self.backward[mu] * src)
            dst += (
                1.0 / 2.0 * (g.gamma[mu] - g.gamma["I"]) * src_plus
                - 1.0 / 2.0 * (g.gamma[mu] + g.gamma["I"]) * src_minus
            )

    def _Mdir(self, dst, src, mu, fb):
        if fb == 1:
            dst @= 1.0 / 2.0 * (g.gamma[mu] - g.gamma["I"]) * self.forward[mu] * src
        elif fb == -1:
            dst @= - 1.0 / 2.0 * (g.gamma[mu] + g.gamma["I"]) * self.backward[mu] * src
        else:
            assert False

    def _Mooee(self, dst, src):
        assert dst != src
        if self.csw is not None:
            dst @= self.clover * src  # clover also contains the diagonal mass term
        else:
            dst @= 1.0 / 2.0 * 1.0 / self.kappa * src

    def _MooeeInv(self, dst, src):
        assert dst != src
        if self.csw is not None:
            dst @= self.clover_inv * src  # clover also contains the diagonal mass term
        else:
            dst @= (2.0 * self.kappa) * src

    def _M(self, dst, src):
        assert dst != src
        dst @= self.Meooe * src + self.Mooee * src
        apply_boundaries(dst, self.open_bc)

    def _G5M(self, dst, src):
        assert dst != src
        dst @= g.gamma[5] * self * src

    def propagator(self, solver):
        exp = self.ExportPhysicalFermionSolution
        imp = self.ImportPhysicalFermionSource

        inv_matrix = solver(self)

        def prop(dst_sc, src_sc):
            g.eval(dst_sc, exp * inv_matrix * imp * src_sc)

        return g.matrix_operator(
            prop,
            otype=(exp.otype[0], imp.otype[1]),
            grid=(exp.grid[0], imp.grid[1]),
            accept_list=True,
        )
