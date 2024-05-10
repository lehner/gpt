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


#
# basis_n_block:  How many basis vectors are cached/moved to accelerator at the same time
#                 A larger number is generally more performant, however, requires more available
#                 cache/accelerator memory.
#
class map:
    def __init__(self, coarse_grid, basis, mask=None, basis_n_block=8):
        assert isinstance(coarse_grid, gpt.grid)
        assert len(basis) > 0

        if mask is None:
            mask = gpt.complex(basis[0].grid)
            mask.checkerboard(basis[0].checkerboard())
            mask[:] = 1
        else:
            assert basis[0].grid is mask.grid
            assert len(mask.v_obj) == 1

        c_otype = gpt.ot_vector_complex_additive_group(len(basis))
        basis_size = c_otype.v_n1[0]
        self.coarse_grid = coarse_grid
        self.basis = basis
        self.obj = cgpt.create_block_map(
            coarse_grid.obj,
            basis,
            basis_size,
            basis_n_block,
            mask.v_obj[0],
        )

        def _project(coarse, fine):
            assert fine[0].checkerboard().__name__ == basis[0].checkerboard().__name__
            cgpt.block_project(self.obj, coarse, fine)

        def _promote(fine, coarse):
            assert fine[0].checkerboard().__name__ == basis[0].checkerboard().__name__
            cgpt.block_promote(self.obj, coarse, fine)

        self.project = gpt.matrix_operator(
            mat=_project,
            adj_mat=_promote,
            vector_space=(
                gpt.vector_space.explicit_grid_otype(coarse_grid, c_otype),
                gpt.vector_space.explicit_lattice(basis[0]),
            ),
            accept_list=True,
        )

        self.promote = gpt.matrix_operator(
            mat=_promote,
            adj_mat=_project,
            vector_space=(
                gpt.vector_space.explicit_lattice(basis[0]),
                gpt.vector_space.explicit_grid_otype(coarse_grid, c_otype),
            ),
            accept_list=True,
        )

    def __del__(self):
        cgpt.delete_block_map(self.obj)

    def orthonormalize(self):
        cgpt.block_orthonormalize(self.obj)

    def check_orthogonality(self, tol=None):
        c_otype = gpt.ot_vector_complex_additive_group(len(self.basis))
        iproj = gpt.lattice(self.coarse_grid, c_otype)
        eproj = gpt.lattice(self.coarse_grid, c_otype)
        for i, v in enumerate(self.basis):
            iproj @= self.project * v
            eproj[:] = 0.0
            eproj[tuple([slice(None, None, None)] * self.coarse_grid.nd + [i])] = 1.0
            err2 = gpt.norm2(eproj - iproj)
            if tol is not None:
                assert err2 <= tol
                gpt.message(f"blockmap: ortho check for vector {i:d}: {err2:e} <= {tol:e}")
            else:
                gpt.message(f"blockmap: ortho check error for vector {i:d}: {err2:e}")

    def coarse_operator(self, fine_operator):
        verbose = gpt.default.is_verbose("block_operator")

        def mat(dst_coarse, src_coarse):
            src_fine = [gpt.lattice(self.basis[0]) for x in src_coarse]
            dst_fine = [gpt.lattice(self.basis[0]) for x in src_coarse]

            t0 = gpt.time()
            self.promote(src_fine, src_coarse)
            t1 = gpt.time()
            fine_operator(dst_fine, src_fine)
            t2 = gpt.time()
            self.project(dst_coarse, dst_fine)
            t3 = gpt.time()
            if verbose:
                gpt.message(
                    "coarse_operator acting on %d vector(s) in %g s (promote %g s, fine_operator %g s, project %g s)"
                    % (len(src_coarse), t3 - t0, t1 - t0, t2 - t1, t3 - t2)
                )

        otype = gpt.ot_vector_complex_additive_group(len(self.basis))
        return gpt.matrix_operator(
            mat=mat,
            vector_space=gpt.vector_space.explicit_grid_otype(self.coarse_grid, otype),
            accept_list=True,
        )

    def fine_operator(self, coarse_operator):
        verbose = gpt.default.is_verbose("block_operator")
        coarse_otype = gpt.ot_vector_complex_additive_group(len(self.basis))

        def mat(dst, src):
            csrc = [gpt.lattice(self.coarse_grid, coarse_otype) for x in src]
            cdst = [gpt.lattice(self.coarse_grid, coarse_otype) for x in src]

            t0 = gpt.time()
            self.project(csrc, src)
            t1 = gpt.time()
            coarse_operator(cdst, csrc)
            t2 = gpt.time()
            self.promote(dst, cdst)
            t3 = gpt.time()
            if verbose:
                gpt.message(
                    "fine_operator acting on %d vector(s) in %g s (project %g s, coarse_operator %g s, promote %g s)"
                    % (len(src), t3 - t0, t1 - t0, t2 - t1, t3 - t2)
                )

        return gpt.matrix_operator(
            mat=mat,
            vector_space=gpt.vector_space.explicit_lattice(self.basis[0]),
            accept_list=True,
        )


# TODO:
# zero -> expects_guess
