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
from gpt.algorithms import base


class checkpointed(base):
    def __init__(self, checkpointer, inverter):
        super().__init__()
        self.inverter = inverter
        self.checkpointer = checkpointer

    def __call__(self, mat):
        vector_space = None
        if isinstance(mat, g.matrix_operator):
            vector_space = mat.vector_space

        inv_mat = self.inverter(mat)
        checkpointer = self.checkpointer

        @self.timed_function
        def inv(psi, src, t):
            checkpointer.grid = psi[0].grid

            t("checkpointer")
            if not checkpointer.load(psi):
                t("inverter")
                inv_mat(psi, src)

                t("checkpointer")
                checkpointer.save(psi)

        return g.matrix_operator(
            mat=inv,
            inv_mat=mat,
            vector_space=vector_space,
            accept_guess=(True, False),
            accept_list=True,
        )
