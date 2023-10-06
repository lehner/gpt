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

# General dense map, add sparse maps


# TODO
# - once happy with design, move time-critical aspects (creation of maps)
#   to cgpt
class map_canonical:
    def __init__(self, n, precision):
        self.n = n
        self.fdimensions = [2**n]
        self.grid = g.grid(self.fdimensions, precision)
        self.verbose = g.default.is_verbose("qis_map")
        self.zero_coordinate = (0,)  # |00000 ... 0> state
        t = g.timer("map_init")
        t("coordinates")
        # TODO: need to split over multiple dimensions, single dimension can hold at most 32 bits
        self.coordinates = g.coordinates(self.grid)
        self.not_coordinates = [np.bitwise_xor(self.coordinates, 2**i) for i in range(n)]
        for i in range(n):
            self.not_coordinates[i].flags["WRITEABLE"] = False
        t("masks")
        self.one_mask = []
        self.zero_mask = []
        for i in range(n):
            proj = np.bitwise_and(self.coordinates, 2**i)

            mask = g.complex(self.grid)
            g.coordinate_mask(mask, proj != 0)
            self.one_mask.append(mask)

            mask = g.complex(self.grid)
            g.coordinate_mask(mask, proj == 0)
            self.zero_mask.append(mask)

        t()
        if self.verbose:
            g.message(t)

    def coordinates_from_permutation(self, bit_permutation):
        ones = np.ones(self.coordinates.shape, self.coordinates.dtype)
        zeros = np.zeros(self.coordinates.shape, self.coordinates.dtype)
        coor = np.copy(zeros)
        for i in range(self.n):
            coor += np.where(
                np.bitwise_and(self.coordinates, 2**i) != 0,
                (2 ** bit_permutation[i]) * ones,
                zeros,
            )
        return coor

    def index_to_bits(self, idx, bit_permutation):
        if bit_permutation is None:
            return [(idx >> shift) & 1 for shift in range(self.n)]
        else:
            return [(idx >> bit_permutation[shift]) & 1 for shift in range(self.n)]

    def bits_to_index(self, bits, bit_permutation):
        if bit_permutation is None:
            return sum([bits[i] << i for i in range(self.n)])
        else:
            return sum([bits[i] << bit_permutation[i] for i in range(self.n)])

    def coordinate_to_basis_name(self, coordinate, bit_permutation=None):
        idx = coordinate[0]
        return (
            "|"
            + ("".join([str(x) for x in reversed(self.index_to_bits(idx, bit_permutation))]))
            + ">"
        )
