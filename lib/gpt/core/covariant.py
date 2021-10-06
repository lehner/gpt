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
import cgpt, gpt, numpy
from gpt.params import params_convention


class shift_base:
    def __init__(self, U, boundary_phases):
        self.nd = len(U)
        self.U = [gpt.copy(u) for u in U]
        self.L = U[0].grid.fdimensions

        if boundary_phases is not None:
            for mu in range(self.nd):
                last_slice = tuple(
                    [
                        self.L[mu] - 1 if mu == nu else slice(None, None, None)
                        for nu in range(self.nd)
                    ]
                )
                self.U[mu][last_slice] = self.U[mu][last_slice] * boundary_phases[mu]

        # now take boundary_phase from params and apply here
        self.Udag = [gpt.eval(gpt.adj(u)) for u in self.U]

        # avoid reference loop
        ref_U = self.U
        ref_Udag = self.Udag

        def _forward(mu):
            def wrap(dst, src):
                dst @= ref_U[mu] * gpt.cshift(src, mu, +1)

            return wrap

        def _backward(mu):
            def wrap(dst, src):
                dst @= gpt.cshift(ref_Udag[mu] * src, mu, -1)

            return wrap

        self.forward = [
            gpt.matrix_operator(mat=_forward(mu), inv_mat=_backward(mu))
            for mu in range(self.nd)
        ]
        self.backward = [o.inv() for o in self.forward]


class shift(shift_base):
    @params_convention(boundary_phases=None)
    def __init__(self, U, params):
        self.params = params
        super().__init__(U, params["boundary_phases"])


class shift_eo(shift):
    @params_convention(boundary_phases=None)
    def __init__(self, U, params):

        # initialize full lattice
        super().__init__(U, params)
        U = None  # do not use anymore / force use including boundary phase

        # create checkerboard version
        self.checkerboard = {}

        # add even/odd functionality
        grid_eo = self.U[0].grid.checkerboarded(gpt.redblack)
        for cb in [gpt.even, gpt.odd]:
            _U = [gpt.lattice(grid_eo, self.U[0].otype) for u in self.U]
            for mu in range(self.nd):
                gpt.pick_checkerboard(cb, _U[mu], self.U[mu])

            self.checkerboard[cb] = shift_base(_U, None)
