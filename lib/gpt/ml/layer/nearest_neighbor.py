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
from gpt.ml.layer import cshift
from gpt.ml.activation import sigmoid


class nearest_neighbor(cshift):
    def __init__(self, grid, ot_input=g.ot_singlet, ot_weights=g.ot_singlet, activation=sigmoid):
        nd = grid.nd
        super().__init__(
            grid,
            ot_input,
            ot_weights,
            [(mu, 1) for mu in range(nd)] + [(mu, -1) for mu in range(nd)],
            activation(grid, ot_input),
        )
