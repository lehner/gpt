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
import numpy as np
from gpt.core.group import differentiable_functional


class inverse_phat_square(differentiable_functional):
    def __init__(self, grid, base, dimensions=None):
        self.base = base

        if dimensions is None:
            dimensions = list(range(grid.nd))

        self.fft = g.fft(dimensions)

        # create FA mask
        cache = {}
        self.weight = g.complex(grid)
        self.weight[:] = 0
        coor = g.coordinates(self.weight)
        for mu in dimensions:
            c_mu = coor[:, mu].astype(np.complex128)
            c_mu_l = g.complex(grid)
            c_mu_l[coor, cache] = c_mu
            c_mu_l @= g.component.sin(c_mu_l * (np.pi / grid.gdimensions[mu]))
            c_mu_l @= c_mu_l * c_mu_l * complex(4.0)
            self.weight += c_mu_l

        # special consideration for zero
        self.weight[0, 0, 0, 0] = (2.0 * np.pi) ** 2.0 / np.prod(
            [grid.gdimensions[mu] for mu in dimensions]
        ) ** (2.0 / len(dimensions))

        # invert
        self.weight @= g.component.inv(self.weight) * complex(4.0 * len(dimensions))
        self.weight = [self.weight]

    @differentiable_functional.multi_field_gradient
    def gradient(self, fields, dfields):
        return g(
            g.inv(self.fft)
            * (self.weight * len(dfields))
            * self.fft
            * self.base.gradient(fields, dfields)
        )

    def __call__(self, *a):
        return self.base(*a)
