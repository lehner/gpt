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
from gpt.ml.layer import sequence as layer_sequence
from gpt.core.group import differentiable_functional


class sequence(layer_sequence):
    def __init__(self, *layers):
        super().__init__(layers)

    # input to output
    def __call__(self, weights, input_layer=None):
        def _mat(dst, src):
            dst @= layer_sequence.__call__(self, weights, src)

        if input_layer is None:
            return g.matrix_operator(mat=_mat)

        return layer_sequence.__call__(self, weights, input_layer)

    def random_weights(self, rng):
        return rng.normal(self.weights())

    def cost(self, training_input, training_output):
        class cost_functional(differentiable_functional):
            def __init__(child):
                child.parent = self

            def __call__(child, weights):
                r = 0.0
                for i, o in zip(training_input, training_output):
                    r += g.norm2(child.parent(weights, i) - o)
                return r

            # d/dx + i d/dy = 2 \partial_{z^*}  for z = x + i y
            # cost = (forward - o)^dag (forward - o)
            # (d/dx + i d/dy)cost = 2.0 * (d/dz^* forward^dag) (forward - o)
            def gradient(child, weights, dweights):
                r = g.group.cartesian(dweights)
                for x in r:
                    x[:] = 0
                for i, o in zip(training_input, training_output):
                    delta = g(2.0 * child.parent(weights, i) - 2.0 * o)
                    gr = child.parent.projected_gradient_adj(weights, i, delta)
                    for nu, dw in enumerate(dweights):
                        mu = weights.index(dw)
                        r[nu] += gr[mu]
                return r

        return cost_functional()
