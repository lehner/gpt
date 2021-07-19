#
#    GPT - Grid Python Toolkit
#    Copyright (C) 2021  Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)
#                  2021  Mattia Bruno
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
from gpt.qcd.fermion.operator.fine_operator import fine_operator


class differentiable_fine_operator(fine_operator):
    def _get_projected_operator(self, functor):
        def _apply(left, right):
            left = gpt.core.util.to_list(left)
            right = gpt.core.util.to_list(right)
            N = len(left)
            nd = len(self.U)
            assert N == len(right)

            ot = self.U[0].otype.cartesian()
            gg = self.U_grid if left[0].grid.cb.n == 1 else self.U_grid_eo
            cb = left[0].checkerboard()
            ders = [gpt.lattice(gg, ot).checkerboard(cb) for _ in range(nd * N)]
            for i in range(N):
                functor(ders[i * nd : (i + 1) * nd], left[i], right[i])

            # change of convention
            for d in ders:
                d @= (1j) * d

            return ders

        return _apply

    def _get_projected_matrix_operator(self, m, md, grid, otype, parity):
        return gpt.projected_matrix_operator(
            self._get_projected_operator(m),
            self._get_projected_operator(md),
            (grid, grid),
            (otype, otype),
            parity,
        )

    def _combined_eooe(self, Meo, Moe):
        def _apply(dst, left, right):
            cb = right.checkerboard()
            if cb is gpt.odd:
                return Meo(dst, left, right)
            else:
                return Moe(dst, left, right)

        return _apply

    def __init__(self, name, U, params, otype=None):
        super().__init__(name, U, params, otype)

        self.M_projected_gradient = self._get_projected_matrix_operator(
            self._MDeriv, self._MDerivDag, self.F_grid, otype, gpt.full
        )
        self.Meooe_projected_gradient = self._get_projected_matrix_operator(
            self._combined_eooe(self._MeoDeriv, self._MoeDeriv),
            self._combined_eooe(self._MeoDerivDag, self._MoeDerivDag),
            self.F_grid_eo,
            otype,
            gpt.odd,
        )
