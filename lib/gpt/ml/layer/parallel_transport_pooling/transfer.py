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

from gpt.ml.layer.parallel_transport_pooling.util import (
    get_fine_gauge_for_paths,
    get_coarse_gauge_for_paths,
)


class transfer:
    def __init__(
        self,
        fine_grid,
        coarse_grid,
        ot_input,
        ot_weights,
        links_and_paths,
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
            get_fine_gauge_for_paths(self.block_transfer, links_and_paths, reference_point)
            if gauge is None
            else gauge
        )
        for x in self.gauge:
            x *= g.norm2(x) ** -0.5
        self.coarse_gauge = (
            get_coarse_gauge_for_paths(self.block_transfer, links_and_paths[0][0], reference_point)
            if coarse_gauge is None
            else coarse_gauge
        )
        self.fine_grid = fine_grid
        self.ot_input = ot_input
        self.ot_weights = ot_weights
        self.n_weights = len(self.gauge)
        self.weights = [g.lattice(fine_grid, ot_weights) for i in range(self.n_weights)]
        self.projector = projector

        if ot_embedding is not None:
            I = g.lattice(self.fine_grid, ot_embedding)
            I = g.identity(I)
            self.gauge = [g(I * x) for x in self.gauge]

    def cloned(self):
        return transfer(
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
        assert len(weights) == len(self.gauge)
        for i in range(self.n_weights):
            x = g(weights[i] * self.gauge[i])
            if i == 0:
                ret = x
            else:
                ret += x
        return ret

    def get_gauge_projected_gradient_adj(self, weights, right, left):
        assert len(weights) == len(self.gauge)
        t = g(right * g.adj(left))
        return [g(self.projector(g.adj(x) * t)) for x in self.gauge]
