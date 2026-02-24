#
#    GPT - Grid Python Toolkit
#    Copyright (C) 2025  Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)
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
from gpt.qcd.scalar.action import fourier_mass_term
from gpt.core.group import differentiable_functional


def assert_real(fields):
    if isinstance(fields, list):
        for f in fields:
            assert_real(f)
    else:
        assert isinstance(fields.otype, g.ot_real_additive_group)


class qed(differentiable_functional):
    def __init__(self, momentum_weight, momentum_mask, name):
        assert_real(momentum_mask)
        self.base = fourier_mass_term(momentum_weight, mask=momentum_mask)
        self.grid = momentum_mask.grid
        self.__name__ = name

    def __call__(self, A):
        assert_real(A)
        return self.base(A)

    def draw(self, A, rng):
        assert_real(A)
        return self.base.draw(A, rng)

    def gradient(self, A, dA):
        assert_real(A)
        assert_real(dA)
        return self.base.gradient(A, dA)

    def propagator(self):
        fourier_mass_field = [
            [g.real(self.grid) for mu in range(self.grid.nd)] for nu in range(self.grid.nd)
        ]
        for i in range(self.grid.nd):
            for j in range(self.grid.nd):
                r = fourier_mass_field[i][j]
                r[:] = 0
                for l in range(self.grid.nd):
                    r += (
                        self.base.fourier_inv_sqrt_inv_mass_field[i][l]
                        * self.base.fourier_inv_sqrt_inv_mass_field[l][j]
                    )

        return [
            [g(g.fft() * self.base.mask * fourier_mass_field[mu][nu]) for nu in range(self.grid.nd)]
            for mu in range(self.grid.nd)
        ]


def qed_l(grid):

    dim = grid.nd
    L = grid.gdimensions
    sqrt_field = [[g.real(grid) for i in range(dim)] for j in range(dim)]

    coor = g.coordinates(sqrt_field[0][0])
    k = [2.0 * np.pi * coor[:, mu] / L[mu] for mu in range(dim)]
    khat = [2.0 * np.sin(k[mu] / 2.0) for mu in range(dim)]

    assert dim == 4  # easy to generalize next two lines
    khatsqr = khat[0] ** 2 + khat[1] ** 2 + khat[2] ** 2 + khat[3] ** 2
    khatsqr_L = khat[0] ** 2 + khat[1] ** 2 + khat[2] ** 2

    sqrt_m = khatsqr**0.5
    eliminated_modes = khatsqr_L < 1e-12
    sqrt_m[eliminated_modes] = 1  # regulator irrelevant, see mask

    mom_mask = g.real(grid)
    mom_mask[:] = 1
    mom_mask[coor[eliminated_modes]] = 0

    for mu in range(dim):
        for nu in range(dim):
            if mu != nu:
                sqrt_field[mu][nu][:] = 0
            else:
                sqrt_field[mu][mu][:] = np.ascontiguousarray(sqrt_m.astype(np.complex128))

    return qed(sqrt_field, mom_mask, "QED_L")
