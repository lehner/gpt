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

    def initialize(self, rng, scale=None):
        pass

    def evaluate(self, inputs, parameters, constants):
        (x,) = inputs
        return [[x] * self.n]


class linear_combination(function):
    """y = x + sum_c w_c X_c with complex numbers w (a residual readout of n
    channels; the identity in x for w = 0).  initialize draws w at
    scale / sqrt(n), so that for similar channels the deviation from the
    identity is of order scale."""

    def __init__(self, template, n, scale=0.01):
        self.n, self.scale = n, scale
        super().__init__([("x", template), ("X", [template] * n)], [("y", template)], [("w", [0j] * n)])

    def initialize(self, rng, scale=None):
        scale = self.scale if scale is None else scale
        self["w"] = [scale / np.sqrt(self.n) * rng.normal_element(0j) / np.sqrt(2) for _ in range(self.n)]

    def evaluate(self, inputs, parameters, constants):
        x, X = inputs
        (w,) = parameters
        return [g(x + sum(wc * xc for wc, xc in zip(w, X)))]


class broadcast(function):
    """y = value 1: a global number as a field (the unit element of the
    template's type times value; Re(value) with real=True, so that the
    parameter stays real).  No inputs.  Its backward sums the flow over the
    sites, so a global parameter can feed functions that take fields."""

    def __init__(self, template, value=0.0, real=False):
        self.value, self.real = value, real
        self.unit = g.identity(template)
        super().__init__([], [("y", template)], [("value", 0j)])

    def initialize(self, rng, scale=None):
        # (a fixed initial value; scale does not apply)
        self["value"] = complex(self.value)

    def evaluate(self, inputs, parameters, constants):
        (value,) = parameters
        if self.real:
            value = g.component.real(value)
        return [g(value * self.unit)]
