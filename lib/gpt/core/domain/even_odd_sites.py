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
import gpt


class even_odd_sites:
    def __init__(self, grid, parity):
        self.checkerboard = parity
        self.grid = grid

    def lattice(self, otype):
        x = gpt.lattice(self.grid, otype)
        x.checkerboard(self.checkerboard)
        return x

    def project(self, dst, src):
        gpt.pick_checkerboard(self.checkerboard, dst, src)

    def promote(self, dst, src):
        gpt.set_checkerboard(dst, src)
