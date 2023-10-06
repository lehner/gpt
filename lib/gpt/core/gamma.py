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
import gpt, cgpt
import numpy as np
from gpt.core.expr import factor
from gpt.core.object_type import ot_matrix_spin

# otype for gamma matrices
gamma_otype = ot_matrix_spin(4)

# basic matrices defining the gamma representation
matrices = {
    0: np.array(
        [[0, 0, 0, 1j], [0, 0, 1j, 0], [0, -1j, 0, 0], [-1j, 0, 0, 0]],
        dtype=np.complex128,
    ),
    1: np.array([[0, 0, 0, -1], [0, 0, 1, 0], [0, 1, 0, 0], [-1, 0, 0, 0]], dtype=np.complex128),
    2: np.array(
        [[0, 0, 1j, 0], [0, 0, 0, -1j], [-1j, 0, 0, 0], [0, 1j, 0, 0]],
        dtype=np.complex128,
    ),
    3: np.array([[0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.complex128),
    4: np.diagflat([1, 1, -1, -1]).astype(dtype=np.complex128),
    11: np.diagflat([1, 1, 1, 1]).astype(dtype=np.complex128),
}


# sigma_xy = 1/2 [gamma_x,gamma_y]
def fill_sigmas():
    idx = 5
    for mu in range(4):
        for nu in range(mu + 1, 4):
            matrices[idx] = 1 / 2 * (matrices[mu] @ matrices[nu] - matrices[nu] @ matrices[mu])
            idx = idx + 1


fill_sigmas()


class gamma_base(factor):
    def __init__(self, gamma):
        self.gamma = gamma
        self.otype = gamma_otype

    def __mul__(self, other):
        if isinstance(other, gpt.tensor):
            return gpt.tensor(
                cgpt.gamma_tensor_mul(other.array, other.otype.v_otype[0], self.gamma, 1),
                other.otype,
            )
        else:
            return super().__mul__(other)

    def __rmul__(self, other):
        if isinstance(other, gpt.tensor):
            return gpt.tensor(
                cgpt.gamma_tensor_mul(other.array, other.otype.v_otype[0], self.gamma, 0),
                other.otype,
            )
        else:
            return super().__rmul__(other)

    def tensor(self):
        assert self.gamma in matrices
        return gpt.tensor(matrices[self.gamma], self.otype)


gamma = {
    0: gamma_base(0),
    1: gamma_base(1),
    2: gamma_base(2),
    3: gamma_base(3),
    5: gamma_base(4),
    "X": gamma_base(0),
    "Y": gamma_base(1),
    "Z": gamma_base(2),
    "T": gamma_base(3),
    "SigmaXY": gamma_base(5),
    "SigmaXZ": gamma_base(6),
    "SigmaXT": gamma_base(7),
    "SigmaYZ": gamma_base(8),
    "SigmaYT": gamma_base(9),
    "SigmaZT": gamma_base(10),
    (0, 1): gamma_base(5),  # other name for Sigma_ij
    (0, 2): gamma_base(6),
    (0, 3): gamma_base(7),
    (1, 2): gamma_base(8),
    (1, 3): gamma_base(9),
    (2, 3): gamma_base(10),
    "I": gamma_base(11),  # identity matrix
}
