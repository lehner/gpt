#
#    GPT - Grid Python Toolkit
#    Copyright (C) 2023  Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)
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
from gpt.ml.layer import base_no_bias


def get_fine_gauge_for_paths(block_transfer, U):
    grid = U[0].grid
    u = g.lattice(U[0])
    u[:] = 0
    pos = g.coordinates(u)
    zero = np.array([0] * grid.nd, dtype=np.int32)

    block_size = [
        grid.gdimensions[i] // block_transfer.coarse_grid.gdimensions[i] for i in range(grid.nd)
    ]
    sparsegrid = pos[(pos % block_size == zero).all(axis=1)]

    for block_pos in np.ndindex(tuple(block_size)):
        if np.all(block_pos == zero):
            mat = g.identity(U[0])
        else:
            path = g.path()
            for i in range(grid.nd):
                path.f(i, block_pos[i])
            pt = g.parallel_transport(U, [path])
            mat = [x for x in pt(U)][0]
        psrc = sparsegrid
        pdst = sparsegrid + np.array(block_pos, dtype=np.int32)
        u[pdst] = mat[psrc]

    return u


def get_coarse_gauge_for_paths(block_transfer, U):
    grid = U[0].grid
    coarse_U = [g.lattice(block_transfer.coarse_grid, u.otype) for u in U]
    pos = g.coordinates(coarse_U[0])

    block_size = np.array(
        [grid.gdimensions[i] // block_transfer.coarse_grid.gdimensions[i] for i in range(grid.nd)],
        dtype=np.int32,
    )

    sparse_pos = block_size * pos

    for mu in range(grid.nd):
        pt = g.parallel_transport(U, [g.path().f(mu, int(block_size[mu]))])
        mat = [x for x in pt(U)][0]
        coarse_U[mu][pos] = mat[sparse_pos]

    return coarse_U


class transfer:
    def __init__(self, fine_grid, coarse_grid, otype, U):
        self.block_transfer = g.block.transfer(fine_grid, coarse_grid, otype)
        self.gauge = get_fine_gauge_for_paths(self.block_transfer, U)
        self.coarse_gauge = get_coarse_gauge_for_paths(self.block_transfer, U)


class project(base_no_bias):
    def __init__(self, transfer):
        self.transfer = transfer
        super().__init__(None, None, None, 0)

    def __call__(self, weights, layer_input):
        return self.transfer.block_transfer.sum(g(self.transfer.gauge * layer_input))

    def projected_gradient_adj(self, weights, layer_input, left):
        return [g(g.adj(self.transfer.gauge) * self.transfer.block_transfer.sum.adj()(left))]


class promote(base_no_bias):
    def __init__(self, transfer):
        self.transfer = transfer
        super().__init__(None, None, None, 0)

    def __call__(self, weights, layer_input):
        return g(g.adj(self.transfer.gauge) * self.transfer.block_transfer.embed(layer_input))

    def projected_gradient_adj(self, weights, layer_input, left):
        return [self.transfer.block_transfer.embed.adj()(self.transfer.gauge * left)]
