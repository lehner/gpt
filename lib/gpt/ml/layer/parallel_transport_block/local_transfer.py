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

import gpt.ml.layer.parallel_transport_block.util as util


class local_transfer:
    def __init__(
        self,
        fine_grid,
        coarse_grid,
        ot_input,
        ot_weights,
        U,
        reference_point=None,
        block_transfer=None,
        gauge=None,
        coarse_gauge=None,
        ot_embedding=None,
        projector=None,
    ):

        if reference_point is None:
            reference_point = np.array([0] * fine_grid.nd, dtype=np.int32)
        else:
            reference_point = np.array(reference_point, dtype=np.int32)

        self.block_transfer = (
            g.block.transfer(fine_grid, coarse_grid, ot_input)
            if block_transfer is None
            else block_transfer
        )
        self.gauge = (
            util.get_fine_gauge_for_paths(self.block_transfer, U, reference_point)
            if gauge is None
            else gauge
        )
        self.coarse_gauge = (
            util.get_coarse_gauge_for_paths(self.block_transfer, U, reference_point)
            if coarse_gauge is None
            else coarse_gauge
        )
        self.fine_grid = fine_grid
        self.ot_input = ot_input
        self.ot_weights = ot_weights
        self.n_weights = 1
        self.weights = [g.lattice(fine_grid, ot_weights)]
        self.projector = projector

        if ot_embedding is not None:
            I = g.lattice(self.fine_grid, ot_embedding)
            I = g.identity(I)
            self.gauge = g(I * self.gauge)

    def clone(self):
        return local_transfer(
            self.fine_grid,
            None,
            self.ot_input,
            self.ot_weights,
            None,
            None,
            block_transfer=self.block_transfer,
            gauge=self.gauge,
            coarse_gauge=self.coarse_gauge,
            ot_embedding=None,
            projector=self.projector,
        )

    def get_gauge(self, weights):
        return g(weights[0] * self.gauge)

    def get_gauge_projected_gradient_adj(self, weights, right, left):
        t = g(g.adj(self.gauge) * right)
        return [g(self.projector(t * g.adj(left)))]
