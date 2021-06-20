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


def inverse_phat_square(grid, df, dimensions=None):
    if dimensions is None:
        dimensions = list(range(grid.nd))

    fft = g.fft(dimensions)

    # create FA mask
    cache = {}
    weight = g.complex(grid)
    weight[:] = 0
    coor = g.coordinates(weight)
    for mu in dimensions:
        c_mu = coor[:, mu].astype(np.complex128)
        c_mu_l = g.complex(grid)
        c_mu_l[coor, cache] = c_mu
        c_mu_l @= g.component.sin(c_mu_l * (np.pi / grid.gdimensions[mu]))
        c_mu_l @= c_mu_l * c_mu_l * complex(4.0)
        weight += c_mu_l

    # special consideration for zero
    weight[0, 0, 0, 0] = (2.0 * np.pi) ** 2.0 / np.prod(
        [grid.gdimensions[mu] for mu in dimensions]
    ) ** (2.0 / len(dimensions))

    # invert
    weight @= g.component.inv(weight) * complex(16.0)

    def df_prime(*arg):
        return g(g.inv(fft) * weight * fft * df(*arg))

    return df_prime
