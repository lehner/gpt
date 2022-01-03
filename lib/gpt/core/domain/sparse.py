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
        self.mask_cache = None

    def exp_ixp(self, mom):
        r = gpt.expr(None)
        for x, p in zip(self.coordinate_lattices(), mom):
            r = r + x * p * 1j
        return gpt.component.exp(r)

    def slice(self, fields, ortho_dim):
        coordinate_obj = self.coordinate_lattices()[ortho_dim].v_obj[0]
        length = self.grid.gdimensions[ortho_dim]
        return gpt.fields_to_tensors(
            fields, lambda src: cgpt.lattice_indexed_sum(src, coordinate_obj, length)
        )

    def mask(self):
        if self.mask_cache is not None:
            return self.mask_cache

        mask = gpt.real(self.grid)
        mask[:] = 0
        mask[self.local_coordinates] = 1

        self.mask_cache = mask
        return mask

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
                np.complex128
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

    def mask(self):
        return self.kernel.mask()

    def coordinate_lattices(self):
        return self.kernel.coordinate_lattices()

    def lattice(self, otype):
        return self.kernel.lattice(otype)

    def exp_ixp(self, mom):
        return self.kernel.exp_ixp(mom)

    def slice(self, fields, ortho_dim):
        return self.kernel.slice(fields, ortho_dim)
