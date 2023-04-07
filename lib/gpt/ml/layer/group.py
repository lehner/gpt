#
#    GPT - Grid Python Toolkit
#    Copyright (C) 2020-22  Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)
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


class group:
    def __init__(self, layers):
        while len(layers) == 1 and isinstance(layers[0], (list, tuple)):
            layers = layers[0]
        self.layers = layers
        self.weights_index = []
        self.weights_list = []
        i0 = 0
        weights_ids = []
        for l in layers:
            w = l.weights()
            if id(w) in weights_ids:
                self.weights_index.append(self.weights_index[weights_ids.index(id(w))])
            else:
                weights_ids.append(id(w))
                self.weights_list = self.weights_list + w
                i1 = i0 + l.n_weights
                self.weights_index.append((i0, i1))
                i0 = i1
        self.n_weights = i0

    def weights(self):
        return self.weights_list

    def forward(self, layer_index, weights, source):
        i0, i1 = self.weights_index[layer_index]
        return self.layers[layer_index](weights[i0:i1], source)

    def dforward_adj(self, layer_index, weights, source, left):
        i0, i1 = self.weights_index[layer_index]
        return self.layers[layer_index].projected_gradient_adj(weights[i0:i1], source, left)
