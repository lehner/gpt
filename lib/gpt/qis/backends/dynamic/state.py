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
#
#    This backend implements bit-permutation ideas similar to De Raedt et al. (2007)
#
#      https://core.ac.uk/download/pdf/205885919.pdf
#
#    to reduce communication overhead.
import gpt as g
import numpy as np
from gpt.qis.map_canonical import map_canonical


class state:
    def __init__(
        self,
        rng,
        number_of_qubits,
        precision=None,
        bit_map=None,
        lattice=None,
        bit_permutation=None,
        current_coordinates=None,
        bit_flipped_plan=None,
    ):
        if precision is None:
            precision = g.double
        if bit_map is None:
            bit_map = map_canonical(number_of_qubits, precision)
        if bit_permutation is None:
            bit_permutation = list(range(number_of_qubits))
            current_coordinates = bit_map.coordinates
        self.rng = rng
        self.precision = precision
        self.number_of_qubits = number_of_qubits
        self.bit_map = bit_map
        self.current_coordinates = current_coordinates
        self.bit_flipped_plan = {} if bit_flipped_plan is None else bit_flipped_plan
        self.bit_permutation = bit_permutation
        self.classical_bit = [None] * number_of_qubits
        if lattice is not None:
            self.lattice = lattice
        else:
            self.lattice = g.complex(self.bit_map.grid)
            self.lattice[:] = 0
            self.lattice[self.bit_map.zero_coordinate] = 1

    def __getitem__(self, idx):
        coor = self.bit_map.bits_to_index(
            self.bit_map.index_to_bits(idx, None), self.bit_permutation
        )
        return self.lattice[(coor,)]

    def cloned(self):
        s = state(
            self.rng,
            self.number_of_qubits,
            self.precision,
            self.bit_map,
            g.copy(self.lattice),
            self.bit_permutation,
            self.current_coordinates,
            self.bit_flipped_plan,
        )
        s.classical_bit = [x for x in self.classical_bit]
        return s

    def prefetch(self, local_qubits):
        high_qubits = []
        for i in range(self.number_of_qubits):
            if i not in local_qubits:
                high_qubits.append(i)
        new_permutation = local_qubits + high_qubits

        # coordinates from bit_permutation
        current_coordinates = self.current_coordinates
        new_coordinates = self.bit_map.coordinates_from_permutation(new_permutation)
        self.lattice[new_coordinates] = self.lattice[current_coordinates]
        self.bit_permutation = new_permutation
        self.current_coordinates = new_coordinates
        self.bit_flipped_plan = {}

    def randomize(self):
        self.rng.cnormal(self.lattice)
        self.lattice *= 1.0 / g.norm2(self.lattice) ** 0.5

    def __str__(self):
        # this is slow but should not be used with large number_of_qubits
        r = ""
        values = self.lattice[self.bit_map.coordinates]
        N_coordinates = len(self.bit_map.coordinates)
        for idx in range(N_coordinates):
            val = values[idx][0]
            if abs(val) > self.precision.eps:
                if len(r) != 0:
                    r += "\n"
                r += (
                    " + "
                    + str(val)
                    + " "
                    + self.bit_map.coordinate_to_basis_name(
                        self.bit_map.coordinates[idx], self.bit_permutation
                    )
                )
        if self.lattice.grid.Nprocessors != 1:
            r += "\n + ..."
        return r

    def bit_flipped_lattice(self, i):
        c = self.bit_map.coordinates
        nci = self.bit_map.not_coordinates[self.bit_permutation[i]]
        # t0 = g.time()
        bfl = g.lattice(self.lattice)
        # t1 = g.time()
        if i not in self.bit_flipped_plan:
            p = g.copy_plan(bfl, self.lattice)
            p.destination += bfl.view[c]
            p.source += self.lattice.view[nci]
            self.bit_flipped_plan[i] = p()
            # g.message(
            #     self.bit_flipped_plan[i].info()
            # )  # TODO: it is odd that this maxes out at 22 GB/s ; focus on bandwidth benchmark first, why 500GB/s for prop and only 5 for singlet?
        # t2 = g.time()
        self.bit_flipped_plan[i](bfl, self.lattice)
        # t3 = g.time()
        # g.message(t1-t0,t2-t1,t3-t2)
        return bfl

    def X(self, i):
        g.copy(self.lattice, self.bit_flipped_lattice(i))

    def R_z(self, i, phi):
        phase_one = np.exp(1j * phi)
        g.bilinear_combination(
            [self.lattice],
            [
                self.bit_map.zero_mask[self.bit_permutation[i]],
                self.bit_map.one_mask[self.bit_permutation[i]],
            ],
            [self.lattice],
            [[1.0, phase_one]],
            [[0, 1]],
            [[0, 0]],
        )

    def H(self, i):
        bfl = self.bit_flipped_lattice(i)
        nrm = 1.0 / 2.0**0.5
        g.bilinear_combination(
            [self.lattice],
            [
                self.bit_map.zero_mask[self.bit_permutation[i]],
                self.bit_map.one_mask[self.bit_permutation[i]],
            ],
            [self.lattice, bfl],
            [[nrm, nrm, -nrm, nrm]],
            [[0, 0, 1, 1]],
            [[0, 1, 0, 1]],
        )

    def CNOT(self, control, target):
        assert control != target
        bfl = self.bit_flipped_lattice(target)
        g.bilinear_combination(
            [self.lattice],
            [
                self.bit_map.zero_mask[self.bit_permutation[control]],
                self.bit_map.one_mask[self.bit_permutation[control]],
            ],
            [self.lattice, bfl],
            [[1.0, 1.0]],
            [[0, 1]],
            [[0, 1]],
        )

    def probability(self, i):
        return g.norm2(self.lattice * self.bit_map.one_mask[self.bit_permutation[i]])

    def measure(self, i):
        p_one = self.probability(i)
        p_zero = 1.0 - p_one
        l = self.rng.uniform_real()
        if l <= p_one:
            proj = self.bit_map.one_mask[self.bit_permutation[i]]
            nrm = 1.0 / (p_one**0.5)
            r = 1
        else:
            proj = self.bit_map.zero_mask[self.bit_permutation[i]]
            nrm = 1.0 / (p_zero**0.5)
            r = 0

        g.bilinear_combination([self.lattice], [proj], [self.lattice], [[nrm]], [[0]], [[0]])

        self.classical_bit[i] = r
        return r


def check_same(state_a, state_b):
    assert (
        np.linalg.norm(
            state_a.lattice[state_a.current_coordinates]
            - state_b.lattice[state_b.current_coordinates]
        )
        < state_a.lattice.grid.precision.eps * 10.0
    )


def check_norm(state):
    assert (g.norm2(state.lattice) - 1.0) < state.lattice.grid.precision.eps * 10.0
