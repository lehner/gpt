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
import sys


class sequence:
    def __init__(self, *inverters):
        self.inverters = inverters

    def __call__(self, outer_mat):

        inverters_mat = [i(outer_mat) for i in self.inverters]

        def inv(dst, src):
            for i in inverters_mat:
                i(dst, src)

        otype, grid, cb = None, None, None
        if isinstance(outer_mat, g.matrix_operator):
            otype, grid, cb = outer_mat.otype, outer_mat.grid, outer_mat.cb

        return g.matrix_operator(
            mat=inv,
            inv_mat=outer_mat,
            otype=otype,
            accept_guess=(True, False),
            grid=grid,
            cb=cb,
            accept_list=True,
        )
