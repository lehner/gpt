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
import gpt as g
from gpt.ml.layer.group import group


class residual(group):
    def __init__(self, *layers, features=None):
        self.features = features
        super().__init__(layers)

    def add(self, a, b):
        return_list = isinstance(a, list)
        a = g.util.to_list(a)
        b = g.util.to_list(b)
        features = self.features
        if features is None:
            features = range(len(a))

        a = g.copy(a)
        for f in features:
            a[f] += b[f]

        if not return_list:
            return a[0]

        return a

    def __call__(self, weights, input_layer):
        current = input_layer
        for i in range(len(self.layers)):
            current = self.forward(i, weights, current)

        return self.add(current, input_layer)

    # out = layer2(w2, layer1(w1, in))
    # left_i partial_i out
    def projected_gradient_adj(self, weights, input_layer, left):
        r = [None for x in weights]
        layer_value = [input_layer]
        # forward propagation
        for i in range(len(self.layers) - 1):
            layer_value.append(self.forward(i, weights, layer_value[-1]))
        # backward propagation
        current_left = left
        for i in reversed(range(len(self.layers))):
            gr = self.dforward_adj(i, weights, layer_value[i], current_left)
            current_left = gr[-1]
            i0, i1 = self.weights_index[i]
            for j in range(i0, i1):
                if r[j] is None:
                    r[j] = gr[j - i0]
                else:
                    r[j] += gr[j - i0]

        return r + [self.add(current_left, left)]
