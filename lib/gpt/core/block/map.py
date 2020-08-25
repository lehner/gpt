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
import gpt, cgpt


class map:
    def __init__(self, coarse_grid, basis, mask=None):
        assert type(coarse_grid) == gpt.grid
        assert len(basis) > 0

        if mask is None:
            mask = gpt.complex(basis[0].grid)
            mask.checkerboard(basis[0].checkerboard())
            mask[:] = 1
        else:
            assert basis[0].grid is mask.grid
            assert len(mask.v_obj) == 1

        c_otype = gpt.ot_vsinglet(len(basis))
        basis_size = c_otype.v_n1[0]
        self.coarse_grid = coarse_grid
        self.basis = basis
        self.obj = cgpt.create_block_map(
            coarse_grid.obj, basis, mask.v_obj[0], len(basis[0].v_obj), basis_size
        )

        def _project(coarse, fine):
            assert fine[0].checkerboard().__name__ == basis[0].checkerboard().__name__
            cgpt.block_project(self.obj, coarse, fine)

        def _promote(fine, coarse):
            assert fine[0].checkerboard().__name__ == basis[0].checkerboard().__name__
            cgpt.block_promote(self.obj, coarse, fine)

        self.project = gpt.matrix_operator(
            mat=_project,
            otype=(c_otype, basis[0].otype),
            grid=(coarse_grid, basis[0].grid),
            cb=(None, basis[0].checkerboard()),
            accept_list=True,
        )

        self.promote = gpt.matrix_operator(
            mat=_promote,
            otype=(basis[0].otype, c_otype),
            grid=(basis[0].grid, coarse_grid),
            cb=(basis[0].checkerboard(), None),
            accept_list=True,
        )

    def __del__(self):
        cgpt.delete_block_map(self.obj)

    def orthonormalize(self):
        cgpt.block_orthonormalize(self.obj)

    def operator(self, op):
        src_fine = gpt.lattice(self.basis[0])
        dst_fine = gpt.lattice(self.basis[0])
        verbose = gpt.default.is_verbose("block_operator")

        def mat(dst_coarse, src_coarse):
            t0 = gpt.time()
            self.promote(src_fine, src_coarse)
            t1 = gpt.time()
            op(dst_fine, src_fine)
            t2 = gpt.time()
            self.project(dst_coarse, dst_fine)
            t3 = gpt.time()
            if verbose:
                gpt.message(
                    "Timing: %g s (promote), %g s (matrix), %g s (project)"
                    % (t1 - t0, t2 - t1, t3 - t2)
                )

        otype = gpt.ot_vsinglet(len(self.basis))

        return gpt.matrix_operator(
            mat=mat, otype=otype, zero=(False, False), grid=self.coarse_grid
        )


# TODO:
# combine otype,grid,cb,n into gpt.vector_space object
# zero -> expects_guess
