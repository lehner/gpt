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
import gpt.ml.layer.parallel_transport_pooling.path
from gpt.ml.layer.parallel_transport_pooling.static_transfer import static_transfer
from gpt.ml.layer.parallel_transport_pooling.transfer import transfer


class project(base_no_bias):
    def __init__(self, t):
        self.transfer = t
        super().__init__(t.fine_grid, t.ot_input, t.ot_weights, t.n_weights)

    def __call__(self, weights, layer_input):
        return self.transfer.block_transfer.sum(g(self.transfer.get_gauge(weights) * layer_input))

    def projected_gradient_adj(self, weights, layer_input, left):
        # layer_input is fine
        # left is coarse
        ret = self.transfer.get_gauge_projected_gradient_adj(
            weights, self.transfer.block_transfer.sum.adj()(left), layer_input
        )
        ret = ret + [
            g(
                g.adj(self.transfer.get_gauge(weights))
                * self.transfer.block_transfer.sum.adj()(left)
            )
        ]
        return ret

    def weights(self):
        return self.transfer.weights


class promote(base_no_bias):
    def __init__(self, t):
        self.transfer = t
        super().__init__(t.fine_grid, t.ot_input, t.ot_weights, t.n_weights)

    def __call__(self, weights, layer_input):
        return g(
            g.adj(self.transfer.get_gauge(weights))
            * self.transfer.block_transfer.embed(layer_input)
        )

    def projected_gradient_adj(self, weights, layer_input, left):
        # layer_input is coarse
        # left is fine
        ret = self.transfer.get_gauge_projected_gradient_adj(
            weights, self.transfer.block_transfer.embed(layer_input), left
        )
        ret = ret + [
            self.transfer.block_transfer.embed.adj()(self.transfer.get_gauge(weights) * left)
        ]
        return ret

    def weights(self):
        return self.transfer.weights
