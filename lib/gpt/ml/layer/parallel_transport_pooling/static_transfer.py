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


class static_transfer:
    def __init__(self, fine_grid, coarse_grid, ot_input, link, path, reference_point=None):
        if reference_point is None:
            reference_point = np.array([0] * fine_grid.nd, dtype=np.int32)
        else:
            reference_point = np.array(reference_point, dtype=np.int32)

        self.block_transfer = g.block.transfer(fine_grid, coarse_grid, ot_input)
        self.gauge = get_fine_gauge_for_paths(self.block_transfer, [(link, path)], reference_point)[
            0
        ]
        self.coarse_gauge = get_coarse_gauge_for_paths(self.block_transfer, link, reference_point)
        self.fine_grid = fine_grid
        self.ot_input = ot_input
        self.ot_weights = None
        self.n_weights = 0
        self.weights = []

    def cloned(self):
        return self

    def get_gauge(self, weights):
        return self.gauge

    def get_gauge_projected_gradient_adj(self, weights, right, left):
        return []
