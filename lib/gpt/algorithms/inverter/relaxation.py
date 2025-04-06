#
#    GPT - Grid Python Toolkit
#    Copyright (C) 2025  Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)
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


class relaxation(base_iterative):
    def __init__(self, inner):
        super().__init__()
        self.inner = inner

    def modified(self, **params):
        return relaxation(self.inner)

    def __call__(self, A):

        vector_space = None
        if isinstance(A, g.matrix_operator):
            vector_space = A.vector_space

        inner_inv = self.inner(A)

        @self.timed_function
        def inv(x, b, t):
            t("matrix")
            r = g(g.expr(b) - A * g.expr(x))
            t("inner")
            g(x, g.expr(x) + inner_inv * g.expr(r))
            t()

        return g.matrix_operator(
            mat=inv,
            inv_mat=A,
            accept_guess=(True, False),
            accept_list=True,
            vector_space=vector_space,
        )
