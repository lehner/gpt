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


class sigmoid:
    def __init__(self, grid, ot_input):
        self.ones = g.lattice(grid, ot_input)
        self.ones[:] = 1

    def __call__(self, x):
        # 1 / (1 + e^-x)
        return g.component.inv(self.ones + g.component.exp(-x))

    def gradient(self, x):
        e = g.component.exp(x)
        return g.component.multiply(e, g.component.pow(-2)(self.ones + e))
