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
from gpt.ml.network import layered
from gpt.core.group import differentiable_functional


class feed_forward(layered):
    def __init__(self, layers):
        super().__init__(layers)

    # input to output
    def __call__(self, weights, input_layer):
        current = input_layer
        for i in range(len(self.layers)):
            current = self.forward(i, weights, current)
        return current

    # out = layer2(w2, layer1(w1, in))
    # left_i partial_i out
    def projected_gradient(self, weights, input_layer, left):
        r = [None for x in weights]
        layer_value = [input_layer]
        # forward propagation
        for i in range(len(self.layers) - 1):
            layer_value.append(self.forward(i, weights, layer_value[-1]))
        # backward propagation
        current_left = left
        for i in reversed(range(len(self.layers))):
            gr = self.dforward(i, weights, layer_value[i], current_left)
            current_left = gr[-1]
            i0, i1 = self.weights_index[i]
            for j in range(i0, i1):
                r[j] = gr[j - i0]
        return r

    def cost(self, training_input, training_output):
        class cost_functional(differentiable_functional):
            def __init__(child):
                child.parent = self

            def __call__(child, weights):
                r = 0.0
                for i, o in zip(training_input, training_output):
                    r += g.norm2(child.parent(weights, i) - o)
                return r

            # cost = (forward - o)^dag (forward - o)
            def gradient(child, weights, dweights):
                r = g.group.cartesian(dweights)
                for x in r:
                    x[:] = 0
                for i, o in zip(training_input, training_output):
                    # next works only if real
                    delta = g(2.0 * child.parent(weights, i) - 2.0 * o)
                    gr = child.parent.projected_gradient(weights, i, delta)
                    for nu, dw in enumerate(dweights):
                        mu = weights.index(dw)
                        r[nu] += gr[mu]
                return r

        return cost_functional()
