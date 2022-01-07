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
import gpt, cgpt
import numpy as np


def get_cache(c, l):
    t = l.otype.__name__
    if t not in c:
        c[t] = {}
    return c[t]


class sparse_kernel:
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
        self.coordinate_lattices_cache = None
        self.weight_cache = None
        self.one_mask_cache = None

    def one_mask(self):
        if self.one_mask_cache is not None:
            return self.one_mask_cache

        ones = gpt.complex(self.embedding_grid)
        ones[:] = 0
        ones[self.embedded_coordinates] = 1
        self.one_mask_cache = ones
        return ones

    def exp_ixp(self, mom, origin):
        r = gpt.expr(None)
        if origin is None:
            for x, p in zip(self.coordinate_lattices(), mom):
                r = r + x * p * 1j
        else:
            for _x, p, _o, _l in zip(
                self.coordinate_lattices(), mom, origin, self.grid.fdimensions
            ):
                lhalf, l, o = int(_l // 2), int(_l), int(_o)
                x = gpt(
                    gpt.component.mod(l)(_x + self.one_mask() * (l + lhalf - o))
                    - lhalf * self.one_mask()
                )
                r = r + x * p * 1j
        return gpt.component.exp(r)

    def slice(self, fields, ortho_dim):
        length = self.grid.gdimensions[ortho_dim]
        return gpt.indexed_sum(fields, self.coordinate_lattices()[ortho_dim], length)

    def weight(self):
        if self.weight_cache is not None:
            return self.weight_cache

        unique_coordinates, count = np.unique(
            self.local_coordinates, axis=0, return_counts=True
        )
        unique_coordinates = unique_coordinates.view(type(self.local_coordinates))
        count = count.astype(self.grid.precision.complex_dtype)

        weight = gpt.complex(self.grid)
        weight[:] = 0
        weight[unique_coordinates] = count

        self.weight_cache = weight
        return weight

    def lattice(self, otype):
        x = gpt.lattice(self.embedding_grid, otype)
        x[:] = 0
        return x

    def coordinate_lattices(self):
        if self.coordinate_lattices_cache is not None:
            return self.coordinate_lattices_cache

        ret = []
        for i in range(self.grid.nd):
            coor_i = self.lattice(gpt.ot_real_additive_group())
            coor_i[self.embedded_coordinates] = self.local_coordinates[:, i].astype(
                self.grid.precision.complex_dtype
            )
            ret.append(coor_i)

        self.coordinate_lattices_cache = ret
        return ret


class sparse:
    def __init__(self, grid, local_coordinates):
        self.local_coordinates = local_coordinates
        self.grid = grid

        # kernel to avoid circular references through captures below
        kernel = sparse_kernel(grid, local_coordinates)

        def _project(dst, src):
            for d, s in zip(dst, src):
                d[kernel.embedded_coordinates, get_cache(kernel.embedded_cache, d)] = s[
                    kernel.local_coordinates, get_cache(kernel.local_cache, s)
                ]

        def _promote(dst, src):
            for d, s in zip(dst, src):
                d[kernel.local_coordinates, get_cache(kernel.local_cache, d)] = s[
                    kernel.embedded_coordinates, get_cache(kernel.embedded_cache, s)
                ]

        self.project = gpt.matrix_operator(
            mat=_project,
            grid=(kernel.embedding_grid, kernel.grid),
            accept_list=True,
            accept_guess=True,
        )

        self.promote = gpt.matrix_operator(
            mat=_promote,
            grid=(kernel.grid, kernel.embedding_grid),
            accept_list=True,
            accept_guess=True,
        )

        self.kernel = kernel

    def weight(self):
        return self.kernel.weight()

    def coordinate_lattices(self):
        return self.kernel.coordinate_lattices()

    def lattice(self, otype):
        return self.kernel.lattice(otype)

    def exp_ixp(self, mom, origin=None):
        return self.kernel.exp_ixp(mom, origin)

    def slice(self, fields, ortho_dim):
        return self.kernel.slice(fields, ortho_dim)
