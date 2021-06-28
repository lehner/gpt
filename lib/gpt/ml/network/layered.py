#
#    GPT - Grid Python Toolkit
#    Copyright (C) 2020  Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)
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


class layered:
    def __init__(self, layers):
        self.layers = layers
        self.weights_index = []
        i0 = 0
        for l in layers:
            i1 = i0 + l.n_weights
            self.weights_index.append((i0, i1))
            i0 = i1

    def random_weights(self, rng):
        w = []
        for l in self.layers:
            w = w + l.weights()
        return rng.normal(w)

    def forward(self, layer_index, weights, source):
        i0, i1 = self.weights_index[layer_index]
        return self.layers[layer_index](weights[i0:i1], source)

    def dforward(self, layer_index, weights, source, left):
        i0, i1 = self.weights_index[layer_index]
        return self.layers[layer_index].projected_gradient(weights[i0:i1], source, left)
