#
#    GPT - Grid Python Toolkit
#    Copyright (C) 2021  Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)
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
import numpy as np


def get_cache(c, l):
    t = l.otype.__name__
    if t not in c:
        c[t] = {}
    return c[t]


class sparse:
    def __init__(self, grid, local_coordinates):
        assert grid.cb.n == 1
        self.grid = grid
        self.local_coordinates = local_coordinates

        # create a minimally embedding lattice geometry
        n = len(local_coordinates)
        N = self.grid.Nprocessors
        l = np.zeros(N, dtype=np.uint64)
        l[self.grid.processor] = 2 ** int(np.ceil(np.log(n) / np.log(2)))
        l = grid.globalsum(l)
        self.L = [int(np.max(l)) * self.grid.mpi[0]] + self.grid.mpi[1:]

        cb_simd_only_first_dimension = gpt.general(
            1, [0] * grid.nd, [1] + [0] * (grid.nd - 1)
        )

        # create grid as subcommunicator so that sparse domains play nice with split grid
        self.embedding_grid = gpt.grid(
            self.L,
            grid.precision,
            cb_simd_only_first_dimension,
            None,
            self.grid.mpi,
            grid,
        )

        self.embedded_coordinates = np.ascontiguousarray(
            gpt.coordinates(self.embedding_grid)[0:n]
        )

        self.embedded_cache = {}
        self.local_cache = {}

    def lattice(self, otype):
        x = gpt.lattice(self.embedding_grid, otype)
        x[:] = 0
        return x

    def project(self, dst, src):
        dst[self.embedded_coordinates, get_cache(self.embedded_cache, dst)] = src[self.local_coordinates, get_cache(self.local_cache, src)]

    def promote(self, dst, src):
        dst[self.local_coordinates, get_cache(self.local_cache, dst)] = src[self.embedded_coordinates, get_cache(self.embedded_cache, src)]

    def coordinate_lattices(self):
        ret = []
        for i in range(self.grid.nd):
            coor_i = gpt.complex(self.embedding_grid)
            coor_i[:] = float("nan")
            coor_i[self.embedded_coordinates] = self.local_coordinates[:, i].astype(
                np.complex128
            )
            ret.append(coor_i)
        return ret
