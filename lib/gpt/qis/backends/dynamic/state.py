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
import gpt as g
import numpy as np



# TODO
# - once happy with design, move time-critical aspects (creation of maps)
#   to cgpt
class map_canonical:
    def __init__(self, n, precision):
        self.n = n
        self.fdimensions = [2 ** n]
        self.grid = g.grid(self.fdimensions, precision)
        self.verbose = g.default.is_verbose("qis_map")
        self.zero_coordinate = (0,)  # |00000 ... 0> state
        t = g.timer("map_init")
        t("coordinates")
        # TODO: need to split over multiple dimensions, single dimension can hold at most 32 bits
        self.coordinates = g.coordinates(self.grid)
        self.not_coordinates = [
            np.bitwise_xor(self.coordinates, 2 ** i) for i in range(n)
        ]
        for i in range(n):
            self.not_coordinates[i].flags["WRITEABLE"] = False
        t("masks")
        self.one_mask = []
        self.zero_mask = []
        for i in range(n):
            proj = np.bitwise_and(self.coordinates, 2 ** i)

            mask = g.complex(self.grid)
            g.coordinate_mask(mask, proj != 0)
            self.one_mask.append(mask)

            mask = g.complex(self.grid)
            g.coordinate_mask(mask, proj == 0)
            self.zero_mask.append(mask)

        t()
        if self.verbose:
            g.message(t)

    def index_to_bits(self, idx):
        return [(idx >> shift) & 1 for shift in range(self.n)]

    def coordinate_to_basis_name(self, coordinate):
        idx = coordinate[0]
        return (
            "|" + ("".join([str(x) for x in reversed(self.index_to_bits(idx))])) + ">"
        )

# IDEAS
# - state should have baseclass serializable, maybe lattice as well
#   use interface in gpt_io
# - X should be virtual, introduce mapping that allows 0<>1 relabeling
#   Concretely, this implements all X and CNOT as changes only to the
#   coordinates (and not_coordinates and masks); all of this would parallelize very well.
#   The only costly part then is the Hadamard.
# - If bit flipped lattice generation is too slow, always evolve all lattices
#   lat, bfl[i] at the same time?  Need more masks etc. but could eliminate
#   cost of bfl?
class state:
    def __init__(
        self, rng, number_of_qubits, precision=None, bit_map=None, lattice=None
    ):
        if precision is None:
            precision = g.double
        if bit_map is None:
            bit_map = map_canonical(number_of_qubits, precision)
        self.rng = rng
        self.precision = precision
        self.number_of_qubits = number_of_qubits
        self.bit_map = bit_map
        self.classical_bit = [None] * number_of_qubits
        if lattice is not None:
            self.lattice = lattice
        else:
            self.lattice = g.complex(self.bit_map.grid)
            self.lattice[:] = 0
            self.lattice[self.bit_map.zero_coordinate] = 1

    def cloned(self):
        s = state(
            self.rng,
            self.number_of_qubits,
            self.precision,
            self.bit_map,
            g.copy(self.lattice),
        )
        s.classical_bit = [x for x in self.classical_bit]
        return s

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
                        self.bit_map.coordinates[idx]
                    )
                )
        if self.lattice.grid.Nprocessors != 1:
            r += "\n + ..."
        return r

    def bit_flipped_lattice(self, i):
        c = self.bit_map.coordinates
        nci = self.bit_map.not_coordinates[i]
        bfl = g.lattice(self.lattice)
        bfl[c] = self.lattice[nci]
        return bfl

    def X(self, i):
        g.copy(self.lattice, self.bit_flipped_lattice(i))

    def R_z(self, i, phi):
        phase_one = np.exp(1j * phi)
        self.lattice @= (
            self.bit_map.zero_mask[i] * self.lattice
            + self.bit_map.one_mask[i] * self.lattice * phase_one
        )

    def H(self, i):
        bfl = self.bit_flipped_lattice(i)
        zero = self.bit_map.zero_mask[i] * self.lattice
        one = self.bit_map.one_mask[i] * self.lattice
        bfl_zero = self.bit_map.one_mask[i] * bfl
        bfl_one = self.bit_map.zero_mask[i] * bfl
        nrm = 1.0 / 2.0 ** 0.5
        self.lattice @= nrm * (zero + bfl_zero) + nrm * (bfl_one - one)

    def CNOT(self, control, target):
        assert control != target
        bfl = self.bit_flipped_lattice(target)
        self.lattice @= (
            self.bit_map.zero_mask[control] * self.lattice
            + self.bit_map.one_mask[control] * bfl
        )

    def measure(self, i):
        p_one = g.norm2(self.lattice * self.bit_map.one_mask[i])
        p_zero = 1.0 - p_one
        l = self.rng.uniform_real()
        if l <= p_one:
            self.lattice @= (self.lattice * self.bit_map.one_mask[i]) / (p_one ** 0.5)
            r = 1
        else:
            self.lattice @= (self.lattice * self.bit_map.zero_mask[i]) / (p_zero ** 0.5)
            r = 0
        self.classical_bit[i] = r
        return r





def check_same(state_a, state_b):
    assert (
        g.norm2(state_a.lattice - state_b.lattice) ** 0.5
        < state_a.lattice.grid.precision.eps
    )


def check_norm(state):
    assert (g.norm2(state.lattice) - 1.0) < state.lattice.grid.precision.eps
