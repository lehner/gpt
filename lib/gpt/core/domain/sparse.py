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
    def __init__(self, grid, unmasked_local_coordinates, dimensions_divisible_by, mask):
        assert grid.cb.n == 1
        self.grid = grid

        if mask is None:
            mask = [True] * len(unmasked_local_coordinates)

        self.local_coordinates = unmasked_local_coordinates[mask]

        if dimensions_divisible_by is None:
            dimensions_divisible_by = [16] + [1] * (self.grid.nd - 1)

        # create a minimally embedding lattice geometry
        n = len(unmasked_local_coordinates)
        N = self.grid.Nprocessors
        l = np.zeros(N, dtype=np.uint64)
        l[self.grid.processor] = n
        l = grid.globalsum(l)
        f_sites = int(np.max(l) * np.prod(self.grid.mpi))
        f_dimensions = np.lcm(self.grid.mpi, dimensions_divisible_by).tolist()
        n_block = int(np.prod(f_dimensions))
        f_dimensions[0] *= (f_sites + n_block - 1) // n_block

        cb_simd_only_first_dimension = gpt.general(1, [0] * grid.nd, [1] + [0] * (grid.nd - 1))

        # create grid as subcommunicator so that sparse domains play nice with split grid
        self.embedding_grid = gpt.grid(
            f_dimensions,
            grid.precision,
            cb_simd_only_first_dimension,
            None,
            self.grid.mpi,
            grid,
        )

        self.embedded_coordinates = np.ascontiguousarray(
            gpt.coordinates(self.embedding_grid)[0 : len(mask)][mask]
        )

        self.embedded_cache = {}
        self.local_cache = {}
        self.coordinate_lattices_cache = {}
        self.one_mask_cache = None
        self.weight_cache = None

    def cached_one_mask(self):
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
                    gpt.component.mod(l)(_x + self.cached_one_mask() * (l + lhalf - o))
                    - lhalf * self.cached_one_mask()
                )
                r = r + x * p * 1j
        return gpt.component.exp(r)

    def slice(self, fields, ortho_dim):
        length = self.grid.gdimensions[ortho_dim]
        return gpt.indexed_sum(fields, self.coordinate_lattices()[ortho_dim], length)

    def weight(self):
        if self.weight_cache is not None:
            return self.weight_cache

        # this function is mostly used for tests, so it is not performance critical
        for rank in range(self.grid.Nprocessors):
            local_coordinates = np.copy(self.local_coordinates)
            self.grid.broadcast(rank, local_coordinates)
            if rank == 0:
                global_coordinates = local_coordinates
            else:
                global_coordinates = np.concatenate((global_coordinates, local_coordinates))

        unique_coordinates, count = np.unique(global_coordinates, axis=0, return_counts=True)
        unique_coordinates = unique_coordinates.view(type(global_coordinates))
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

    def coordinate_lattices(self, mark_empty=0):
        if mark_empty in self.coordinate_lattices_cache:
            return self.coordinate_lattices_cache[mark_empty]

        ret = []
        for i in range(self.grid.nd):
            coor_i = self.lattice(gpt.ot_real_additive_group())
            coor_i[:] = mark_empty
            coor_i[self.embedded_coordinates] = self.local_coordinates[:, i].astype(
                self.grid.precision.complex_dtype
            )
            ret.append(coor_i)

        self.coordinate_lattices_cache[mark_empty] = ret
        return ret


class sparse:
    def __init__(self, grid, unmasked_local_coordinates, dimensions_divisible_by=None, mask=None):
        self.grid = grid

        # kernel to avoid circular references through captures below
        kernel = sparse_kernel(grid, unmasked_local_coordinates, dimensions_divisible_by, mask)

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
            vector_space=(
                gpt.vector_space.explicit_grid(kernel.embedding_grid),
                gpt.vector_space.explicit_grid(kernel.grid),
            ),
            accept_list=True,
            accept_guess=True,
        )

        self.promote = gpt.matrix_operator(
            mat=_promote,
            vector_space=(
                gpt.vector_space.explicit_grid(kernel.grid),
                gpt.vector_space.explicit_grid(kernel.embedding_grid),
            ),
            accept_list=True,
            accept_guess=True,
        )

        self.kernel = kernel
        self.local_coordinates = kernel.local_coordinates

    def converted(self, precision):
        if self.kernel.grid.precision is precision:
            return self

        grid_embed = self.kernel.grid.converted(precision)
        cl = self.coordinate_lattices(mark_empty=-1)
        mask = (cl[0][:] >= 0)[:, 0]
        local_coordinates = np.hstack(tuple([x[:].real.astype(np.int32) for x in cl]))

        return sparse(
            grid_embed,
            local_coordinates,
            dimensions_divisible_by=cl[0].grid.fdimensions,
            mask=mask,
        )

    def weight(self):
        return self.kernel.weight()

    def unique_embedded_coordinates(self, coordinates):
        idx1 = self.grid.lexicographic_index(coordinates)
        idx2 = self.grid.lexicographic_index(self.kernel.local_coordinates)

        # all nodes first in parallel create lists of coordinates they can offer locally
        m = {}
        for i, j in enumerate(idx2):
            ec = self.kernel.embedded_coordinates[i, :]
            ij = int(j)
            if ij not in m:
                m[ij] = [ec]
            else:
                m[ij].append(ec)

        # now create a map that we fill in with information from one rank at a time
        mp = np.full(
            shape=(len(coordinates) + 1, self.kernel.embedding_grid.nd),
            fill_value=-1,
            dtype=np.int32,
        )
        mp[-1, 0] = len(coordinates)
        for rank in range(self.kernel.embedding_grid.Nprocessors):
            if rank == self.kernel.embedding_grid.processor:
                left = mp[-1, 0]
                for i in range(len(coordinates)):
                    idx1_i = idx1[i]
                    if mp[i, 0] == -1 and idx1_i in m:
                        mm = m[idx1_i]
                        if len(mm) > 0:
                            mp[i] = mm.pop()
                            left -= 1
                mp[-1, 0] = left
            self.kernel.embedding_grid.broadcast(rank, mp)
            if mp[-1, 0] == 0:
                break

        return np.ascontiguousarray(mp[0:-1])

    def coordinate_lattices(self, **args):
        return self.kernel.coordinate_lattices(**args)

    def lattice(self, otype):
        return self.kernel.lattice(otype)

    def exp_ixp(self, mom, origin=None):
        return self.kernel.exp_ixp(mom, origin)

    def slice(self, fields, ortho_dim):
        return self.kernel.slice(fields, ortho_dim)

    def conformable(self, other):
        # conformable up to lattice precision
        a = self.coordinate_lattices(mark_empty=-1)
        b = other.coordinate_lattices(mark_empty=-1)
        if len(a) != len(b):
            return False
        for mu in range(len(a)):
            eps2 = gpt.norm2(gpt.convert(b[mu], gpt.double) - gpt.convert(a[mu], gpt.double))
            if eps2 > 1e-13:
                return False
        return True
