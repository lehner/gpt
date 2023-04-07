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


def get_fine_gauge_for_paths(block_transfer, links_and_paths, reference_point):
    grid = links_and_paths[0][0][0].grid
    u = g.lattice(links_and_paths[0][0][0])
    u[:] = 0
    pos = g.coordinates(u)

    block_size = [
        grid.gdimensions[i] // block_transfer.coarse_grid.gdimensions[i] for i in range(grid.nd)
    ]

    zero = np.array([0] * grid.nd, dtype=np.int32)
    sparsegrid = pos[(pos % block_size == zero).all(axis=1)]

    ret = []
    for lp in links_and_paths:
        if len(lp) == 2:
            U, get_path = lp
            V = g.identity(u)
        else:
            U, get_path, V = lp

        for block_pos in np.ndindex(tuple(block_size)):
            if np.all(block_pos == reference_point):
                mat = g.identity(U[0])
            else:
                pt = g.parallel_transport(U, [get_path(block_pos - reference_point)])
                mat = [x for x in pt(U)][0]
            psrc = sparsegrid + reference_point
            pdst = sparsegrid + np.array(block_pos, dtype=np.int32)
            u[pdst] = mat[psrc]
        ret.append(g(u * g.adj(V)))

    return ret


def get_coarse_gauge_for_paths(block_transfer, U, reference_point):
    grid = U[0].grid
    coarse_U = [g.lattice(block_transfer.coarse_grid, u.otype) for u in U]
    pos = g.coordinates(coarse_U[0])

    block_size = np.array(
        [grid.gdimensions[i] // block_transfer.coarse_grid.gdimensions[i] for i in range(grid.nd)],
        dtype=np.int32,
    )

    sparse_pos = block_size * pos + reference_point

    for mu in range(grid.nd):
        pt = g.parallel_transport(U, [g.path().f(mu, int(block_size[mu]))])
        mat = [x for x in pt(U)][0]
        coarse_U[mu][pos] = mat[sparse_pos]

    return coarse_U
