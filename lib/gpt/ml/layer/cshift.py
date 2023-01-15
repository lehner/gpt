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
from gpt.ml.layer import base


class cshift(base):
    def __init__(self, grid, ot_input, ot_weights, shifts, activation):
        super().__init__(grid, ot_input, ot_weights, len(shifts) + 2)
        self.shifts = shifts
        self.ones = g.lattice(grid, ot_input)
        self.ones[:] = 1
        self.activation = activation

    def shift(self, x, i):
        if i == 0:
            return self.ones
        elif i == 1:
            return x

        d, s = self.shifts[i - 2]
        return g.cshift(x, d, s)

    def ishift(self, x, i):
        if i == 0:
            return self.ones
        elif i == 1:
            return x

        d, s = self.shifts[i - 2]
        return g.cshift(x, d, -s)

    # out_{i} = activation(w[0]_{i} + w[1]_{il} * shift[1,in]_{l} + ...)
    def __call__(self, weights, layer_input):
        assert len(weights) == self.n_weights
        s = g.expr(weights[0])
        for i in range(1, self.n_weights):
            s += weights[i] * self.shift(layer_input, i)
        return self.activation(g(s))

    # pg_{w_{ab}} = left_j partial_{w_{ab}} out_j
    # = activation'(...)_a left_a shift[1,in]_b
    # pg_{in_a} = left_j partial_{in_a} out_j
    # = activation'(...)_j left_j ishift[1,w[1]_{ja}]
    def projected_gradient_adj(self, weights, layer_input, left):
        assert len(weights) == self.n_weights
        shifts = [self.shift(layer_input, j) for j in range(self.n_weights)]

        s = g.expr(weights[0])
        for w, sh in zip(weights[1:], shifts[1:]):
            s += w * sh

        dactivation = self.activation.gradient(g(s))
        left_dactivation = g.component.multiply(left, g.conj(dactivation))

        dinput = g.adj(weights[1]) * left_dactivation
        for i in range(2, len(weights)):
            dinput += g(self.ishift(g(g.adj(weights[i]) * left_dactivation), i))

        r = [left_dactivation]
        for i in range(1, self.n_weights):
            o = g.group.cartesian(weights[i])
            o @= left_dactivation * g.adj(shifts[i])
            r.append(o)
        r.append(g(dinput))
        return r
