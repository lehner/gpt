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
import gpt as g
import numpy as np
from gpt.params import params_convention
from gpt.core.covariant import shift_eo
from gpt import matrix_operator


class wilson_clover(shift_eo, matrix_operator):
    # M = sum_mu gamma[mu]*D[mu] + m0 - 1/2 sum_mu D^2[mu]
    # m0 + 4 = 1/2/kappa
    @params_convention(
        kappa=None,
        mass=None,
        cF=1,
        use_legacy=False,
        boundary_phases=None,
        xi_0=None,
        csw_r=None,
        csw_t=None,
        nu=None,
        isAnisotropic=None,
    )
    def __init__(self, U, params):
        shift_eo.__init__(self, U, boundary_phases=params["boundary_phases"])

        Nc = U[0].otype.Nc
        otype = g.ot_vector_spin_color(4, Nc)
        grid = U[0].grid
        grid_eo = grid.checkerboarded(g.redblack)
        self.F_grid = grid
        self.U_grid = grid
        self.F_grid_eo = grid_eo
        self.U_grid_eo = grid_eo

        self.vector_space_F = g.vector_space.explicit_grid_otype(self.F_grid, otype)
        self.vector_space_U = g.vector_space.explicit_grid_otype(self.U_grid, otype)
        self.vector_space_F_eo = g.vector_space.explicit_grid_otype(self.F_grid_eo, otype)

        self.src_e = g.vspincolor(grid_eo)
        self.src_o = g.vspincolor(grid_eo)
        self.dst_e = g.vspincolor(grid_eo)
        self.dst_o = g.vspincolor(grid_eo)
        self.dst_e.checkerboard(g.even)
        self.dst_o.checkerboard(g.odd)

        if params["kappa"] is not None:
            assert params["mass"] is None
            self.m0 = 1.0 / params["kappa"] / 2.0 - 4.0
        else:
            self.m0 = params["mass"]

        self.xi_0 = params["xi_0"]
        self.csw_r = params["csw_r"] / self.xi_0
        self.csw_t = params["csw_t"]
        self.nu = params["nu"]

        self.kappa = 1.0 / (2.0 * (self.m0 + 1.0 + 3.0 * self.nu / self.xi_0))

        self.open_bc = params["boundary_phases"][self.nd - 1] == 0.0
        if self.open_bc:
            assert all(
                [
                    self.xi_0 == 1.0,
                    self.nu == 1.0,
                    self.csw_r == self.csw_t,
                    "cF" in params,
                ]
            )  # open bc only for isotropic case, require cF passed
            self.cF = params["cF"]
            T = self.L[self.nd - 1]

        # compute field strength tensor
        if self.csw_r != 0.0 or self.csw_t != 0.0:
            self.clover = g.mspincolor(grid)
            self.clover[:] = 0
            I = g.identity(self.clover)
            for mu in range(self.nd):
                for nu in range(mu + 1, self.nd):
                    if mu == (self.nd - 1) or nu == (self.nd - 1):
                        cp = self.csw_t
                    else:
                        cp = self.csw_r
                    self.clover += (
                        -0.5 * cp * g.gamma[mu, nu] * I * g.qcd.gauge.field_strength(U, mu, nu)
                    )

            if self.open_bc:
                # set field strength tensor to unity at the temporal boundaries
                value = -0.5 * self.csw_t
                self.clover[:, :, :, 0, :, :, :, :] = 0.0
                self.clover[:, :, :, T - 1, :, :, :, :] = 0.0
                for alpha in range(4):
                    for a in range(Nc):
                        self.clover[:, :, :, 0, alpha, alpha, a, a] = value
                        self.clover[:, :, :, T - 1, alpha, alpha, a, a] = value

                if self.cF != 1.0:
                    # add improvement coefficients next to temporal boundaries
                    value = self.cF - 1.0
                    for alpha in range(4):
                        for a in range(Nc):
                            self.clover[:, :, :, 1, alpha, alpha, a, a] += value
                            self.clover[:, :, :, T - 2, alpha, alpha, a, a] += value

            # integrate kappa into clover matrix for inversion
            self.clover += 1.0 / 2.0 * 1.0 / self.kappa * I

            self.clover_inv = g.matrix.inv(self.clover)

            self.clover_eo = {
                g.even: g.lattice(grid_eo, self.clover.otype),
                g.odd: g.lattice(grid_eo, self.clover.otype),
            }
            self.clover_inv_eo = {
                g.even: g.lattice(grid_eo, self.clover.otype),
                g.odd: g.lattice(grid_eo, self.clover.otype),
            }
            for cb in self.clover_eo:
                g.pick_checkerboard(cb, self.clover_eo[cb], self.clover)
                g.pick_checkerboard(cb, self.clover_inv_eo[cb], self.clover_inv)
        else:
            self.clover = None
            self.clover_inv = None

        self.Meooe = g.matrix_operator(
            mat=lambda dst, src: self._Meooe(dst, src),
            vector_space=self.vector_space_F_eo,
        )
        self.Mooee = g.matrix_operator(
            mat=lambda dst, src: self._Mooee(dst, src),
            inv_mat=lambda dst, src: self._MooeeInv(dst, src),
            vector_space=self.vector_space_F_eo,
        )
        self.Dhop = g.matrix_operator(
            mat=lambda dst, src: self._Dhop(dst, src), vector_space=self.vector_space_F
        )
        matrix_operator.__init__(
            self, lambda dst, src: self._M(dst, src), vector_space=self.vector_space_F
        )
        self.G5M = g.matrix_operator(
            lambda dst, src: self._G5M(dst, src), vector_space=self.vector_space_F
        )
        self.Mdiag = g.matrix_operator(
            lambda dst, src: self._Mdiag(dst, src), vector_space=self.vector_space_F
        )
        self.ImportPhysicalFermionSource = g.matrix_operator(
            lambda dst, src: g.copy(dst, src), vector_space=self.vector_space_F
        )
        self.ExportPhysicalFermionSolution = g.matrix_operator(
            lambda dst, src: g.copy(dst, src), vector_space=self.vector_space_F
        )
        self.ExportPhysicalFermionSource = g.matrix_operator(
            lambda dst, src: g.copy(dst, src), vector_space=self.vector_space_F
        )
        self.Dminus = g.matrix_operator(
            lambda dst, src: g.copy(dst, src), vector_space=self.vector_space_F
        )

    def apply_boundaries(self, dst):
        if self.open_bc:
            g.qcd.fermion.apply_open_boundaries(dst)

    def _Meooe(self, dst, src):
        assert dst != src
        cb = src.checkerboard()
        scb = self.checkerboard[cb]
        scbi = self.checkerboard[cb.inv()]
        dst.checkerboard(cb.inv())
        dst[:] = 0
        for mu in range(self.nd):
            src_plus = g.eval(scbi.forward[mu] * src)
            src_minus = g.eval(scb.backward[mu] * src)
            if mu == self.nd - 1:
                cc = 1.0
            else:
                cc = self.nu / self.xi_0
            dst += (
                cc / 2.0 * (g.gamma[mu] - g.gamma["I"]) * src_plus
                - cc / 2.0 * (g.gamma[mu] + g.gamma["I"]) * src_minus
            )
        self.apply_boundaries(dst)

    def _Mooee(self, dst, src):
        assert dst != src
        cb = src.checkerboard()
        dst.checkerboard(cb)
        if self.clover is not None:
            dst @= self.clover_eo[cb] * src
        else:
            dst @= 1.0 / 2.0 * 1.0 / self.kappa * src
        self.apply_boundaries(dst)

    def _MooeeInv(self, dst, src):
        assert dst != src
        cb = src.checkerboard()
        dst.checkerboard(cb)
        if self.clover_inv is not None:
            dst @= self.clover_inv_eo[cb] * src
        else:
            dst @= 2.0 * self.kappa * src
        self.apply_boundaries(dst)

    def _Dhop(self, dst, src):
        assert dst != src

        g.pick_checkerboard(g.even, self.src_e, src)
        g.pick_checkerboard(g.odd, self.src_o, src)

        self.dst_o @= self.Meooe * self.src_e
        self.dst_e @= self.Meooe * self.src_o

        g.set_checkerboard(dst, self.dst_o)
        g.set_checkerboard(dst, self.dst_e)

        self.apply_boundaries(dst)

    def _M(self, dst, src):
        assert dst != src

        g.pick_checkerboard(g.even, self.src_e, src)
        g.pick_checkerboard(g.odd, self.src_o, src)

        self.dst_o @= self.Meooe * self.src_e + self.Mooee * self.src_o
        self.dst_e @= self.Meooe * self.src_o + self.Mooee * self.src_e

        g.set_checkerboard(dst, self.dst_o)
        g.set_checkerboard(dst, self.dst_e)

        self.apply_boundaries(dst)

    def _Mdiag(self, dst, src):
        assert dst != src

        g.pick_checkerboard(g.even, self.src_e, src)
        g.pick_checkerboard(g.odd, self.src_o, src)

        self.dst_o @= self.Mooee * self.src_o
        self.dst_e @= self.Mooee * self.src_e

        g.set_checkerboard(dst, self.dst_o)
        g.set_checkerboard(dst, self.dst_e)

        self.apply_boundaries(dst)

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
            vector_space=self.vector_space_U,
            accept_list=True,
        )
