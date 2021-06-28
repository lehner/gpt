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
    def __init__(self, grid, shifts, activation):
        super().__init__(grid, len(shifts) + 2)
        self.shifts = shifts
        self.ones = g.complex(grid)
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

    def __call__(self, weights, layer_input):
        assert len(weights) == self.n_weights
        s = g.expr(weights[0])
        for i in range(1, self.n_weights):
            s += weights[i] * self.shift(layer_input, i)
        return self.activation(s)

    def projected_gradient(self, weights, layer_input, left):
        assert len(weights) == self.n_weights
        shifts = [self.shift(layer_input, j) for j in range(self.n_weights)]

        s = weights[0] * shifts[0]
        for w, sh in zip(weights[1:], shifts[1:]):
            s += w * sh

        dactivation = self.activation.gradient(s)
        left_dactivation = g(left * dactivation)

        dinput = left_dactivation * weights[1]
        for i in range(2, len(weights)):
            dinput += g(self.ishift(g(left_dactivation * weights[i]), i))
        return [g(left_dactivation * shifts[i]) for i in range(self.n_weights)] + [
            g(dinput)
        ]
