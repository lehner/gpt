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
from gpt.params import params_convention


class coarse_deflate:
    @params_convention(block=32, linear_combination_block=4, fine_block=8)
    def __init__(self, cevec, basis, fev, params):
        self.mat = g.block.map(
            cevec[0].grid, basis, basis_n_block=params["fine_block"]
        ).fine_operator(
            g.algorithms.modes.matrix(
                cevec,
                cevec,
                fev,
                lambda x: 1.0 / x,
                block=params["block"],
                linear_combination_block=params["linear_combination_block"],
            )
        )

    def __call__(self, matrix):
        return self.mat
