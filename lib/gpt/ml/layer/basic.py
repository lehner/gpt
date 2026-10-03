#
#    GPT - Grid Python Toolkit
#    Copyright (C) 2026  Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)
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
from gpt.ml.function import function


class replicate(function):
    """X = [x] * n: a field as n channels."""

    def __init__(self, template, n):
        self.n = n
        super().__init__([("x", template)], [("X", [template] * n)], [])

    def initialize(self, rng):
        pass

    def evaluate(self, inputs, parameters, constants):
        (x,) = inputs
        return [[x] * self.n]


class linear_combination(function):
    """y = x + sum_c w_c X_c with complex numbers w (a residual readout of n
    channels)."""

    def __init__(self, template, n, scale=0.01):
        self.n, self.scale = n, scale
        super().__init__([("x", template), ("X", [template] * n)], [("y", template)], [("w", [0j] * n)])

    def initialize(self, rng):
        # small, not zero: with zero weights, earlier functions get no gradient
        self["w"] = [self.scale * rng.normal_element(0j) / np.sqrt(2) for _ in range(self.n)]

    def evaluate(self, inputs, parameters, constants):
        x, X = inputs
        (w,) = parameters
        return [g(x + sum(wc * xc for wc, xc in zip(w, X)))]
