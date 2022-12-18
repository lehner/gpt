#
#    GPT - Grid Python Toolkit
#    Copyright (C) 2022  Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)
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
from gpt.algorithms import base_iterative


class multi_shift(base_iterative):
    def __init__(self, inverter, shifts):
        self.inverter = inverter
        self.shifts = shifts
        super().__init__()

    def __call__(self, mat):
        def _mat(dst, src, s_val):
            g.eval(dst, mat * src)
            for d, s in zip(dst, src):
                d += s_val * s

        inverter_mat = [
            self.inverter(
                g.matrix_operator(
                    mat=lambda dst, src, s_val=s: _mat(dst, src, s_val),
                    accept_list=True,
                )
            )
            for s in self.shifts
        ]

        @self.timed_function
        def inv(dst, src, t):
            for j, i in enumerate(inverter_mat):
                i(dst[j * len(src) : (j + 1) * len(src)], src)

        vector_space = None
        if isinstance(mat, g.matrix_operator):
            vector_space = mat.vector_space

        return g.matrix_operator(
            mat=inv,
            vector_space=vector_space,
            accept_guess=(True, False),
            accept_list=lambda src: len(src) * len(self.shifts),
        )
