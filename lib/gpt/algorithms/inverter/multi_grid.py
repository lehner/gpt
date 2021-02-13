#
#    GPT - Grid Python Toolkit
#    Copyright (C) 2020  Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)
#                  2020  Daniel Richtmann (daniel.richtmann@ur.de)
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


class multi_grid_setup:
    @g.params_convention(block_size=None, projector=None)
    def __init__(self, params):
        self.block_size = params["block_size"]
        self.n = len(self.block_size)
        self.projector = g.util.to_list(params["projector"], self.n)

    def __call__(self, matrix):
        levels = []
        grid = matrix.grid[0]
        for i in range(self.n):
            grid = g.block.grid(grid, self.block_size[i])
            basis = self.projector[i](matrix, grid)
            levels.append((grid, basis))
            if i != self.n - 1:
                matrix = matrix.coarsened(*levels[-1])
        return levels


class multi_grid:
    @g.params_convention(make_hermitian=False, save_links=True)
    def __init__(self, coarse_inverter, coarse_grid, basis, params):
        self.params = params
        self.coarse_inverter = coarse_inverter
        self.coarse_grid = coarse_grid
        self.verbose = g.default.is_verbose("multi_grid_inverter")
        self.basis = basis

    def __call__(self, mat):

        assert isinstance(mat, g.matrix_operator)
        otype, fine_grid, cb = mat.otype, mat.grid, mat.cb

        bm = g.block.map(self.coarse_grid, self.basis)

        cmat = mat.coarsened(self.coarse_grid, self.basis)

        cinv = self.coarse_inverter(cmat)

        def inv(dst, src):
            assert dst != src
            if self.verbose:
                g.message(f"Enter grid {self.coarse_grid}")
                t0 = g.time()

            g.eval(dst, bm.promote * cinv * bm.project * src)

            if self.verbose:
                t1 = g.time()
                g.message(
                    f"Back to grid {fine_grid[0]}, spent {t1-t0} seconds on coarse grid"
                )

        return g.matrix_operator(
            mat=inv,
            inv_mat=mat,
            adj_mat=None,
            adj_inv_mat=None,
            otype=otype,
            accept_guess=(True, False),
            accept_list=True,
            grid=fine_grid,
            cb=cb,
        )
