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


class split:
    @g.params_convention(mpi_split=None)
    def __init__(self, operation, params):
        self.params = params
        self.operation = operation

    def __call__(self, matrix):
        mpi_split = self.params["mpi_split"]
        matrix_split = matrix.split(mpi_split)
        operation_split = self.operation(matrix_split)
        nparallel = matrix_split.vector_space[0].grid.sranks
        cache = {}

        def inv(dst, src):
            # verbosity
            verbose = g.default.is_verbose("split")

            if len(src) % nparallel != 0:
                raise Exception(f"Cannot divide {len(src)} global vectors into {nparallel} groups")

            t0 = g.time()
            src_split = g.split(src, matrix_split.vector_space[1].grid, cache)
            dst_split = g.split(dst, matrix_split.vector_space[0].grid, cache)
            t1 = g.time()

            operation_split(dst_split, src_split)

            t2 = g.time()
            g.unsplit(dst, dst_split, cache)
            t3 = g.time()

            if verbose:
                g.message(
                    f"Split {len(src)} global vectors to {len(src_split)} local vectors\n"
                    + f"Timing: {t1-t0} s (split), {t2-t1} s (operation), {t3-t2} s (unsplit)"
                )

        vector_space = None
        if isinstance(matrix, g.matrix_operator):
            vector_space = matrix.vector_space

        return g.matrix_operator(
            mat=inv,
            inv_mat=matrix,
            accept_guess=(True, False),
            vector_space=vector_space,
            accept_list=True,
        )
