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
import gpt, cgpt, sys
from gpt.core.block.operator import operator


def grid(fgrid, nblock):
    assert fgrid.nd == len(nblock)
    for i in range(fgrid.nd):
        assert fgrid.fdimensions[i] % nblock[i] == 0
    # coarse grid will always be a full grid
    return gpt.grid(
        [fgrid.fdimensions[i] // nblock[i] for i in range(fgrid.nd)],
        fgrid.precision,
        gpt.full,
        parent=fgrid.parent,
    )


def project(coarse, fine, basis):
    assert fine.checkerboard().__name__ == basis[0].checkerboard().__name__
    cot = coarse.otype
    fot = fine.otype
    tmp = gpt.lattice(coarse)
    coarse[:] = 0
    for j in fot.v_idx:
        for i in cot.v_idx:
            cgpt.block_project(
                tmp.v_obj[i], fine.v_obj[j], basis[cot.v_n0[i] : cot.v_n1[i]], j
            )
        coarse += tmp
    return coarse


def promote(coarse, fine, basis):
    assert len(basis) > 0
    cot = coarse.otype
    fot = fine.otype
    fine.checkerboard(basis[0].checkerboard())
    tmp = gpt.lattice(fine)
    fine[:] = 0
    for i in cot.v_idx:
        for j in fot.v_idx:
            cgpt.block_promote(
                coarse.v_obj[i], tmp.v_obj[j], basis[cot.v_n0[i] : cot.v_n1[i]], j
            )
        fine += tmp
    return fine


def maskedInnerProduct(coarse, fineMask, fineX, fineY):
    assert fineX.checkerboard().__name__ == fineY.checkerboard().__name__
    assert fineX.otype.__name__ == fineY.otype.__name__
    assert len(coarse.v_obj) == 1
    assert len(fineMask.v_obj) == 1
    fot = fineX.otype
    tmp = gpt.lattice(coarse)
    coarse[:] = 0
    for i in fot.v_idx:
        cgpt.block_maskedInnerProduct(
            tmp.v_obj[0], fineMask.v_obj[0], fineX.v_obj[i], fineY.v_obj[i]
        )
        coarse += tmp
    return coarse


def innerProduct(coarse, fineX, fineY):
    assert fineX.checkerboard().__name__ == fineY.checkerboard().__name__
    assert fineX.otype.__name__ == fineY.otype.__name__
    assert len(coarse.v_obj) == 1
    cgpt.block_innerProduct(coarse.v_obj[0], fineX, fineY)
    return coarse


def innerProduct_other(coarse, fineX, fineY):
    assert fineX.checkerboard().__name__ == fineY.checkerboard().__name__
    assert fineX.otype.__name__ == fineY.otype.__name__
    assert len(coarse.v_obj) == 1
    fot = fineX.otype
    tmp = gpt.lattice(coarse)
    coarse[:] = 0
    for i in fot.v_idx:
        cgpt.block_innerProduct_test(tmp.v_obj[0], fineX.v_obj[i], fineY.v_obj[i])
        coarse += tmp
    return coarse


def zaxpy(fineZ, coarseA, fineX, fineY):
    assert fineX.checkerboard().__name__ == fineY.checkerboard().__name__
    assert fineX.otype.__name__ == fineY.otype.__name__ == fineZ.otype.__name__
    assert len(coarseA.v_obj) == 1
    fineZ.checkerboard(fineX.checkerboard())
    fot = fineX.otype
    for i in fot.v_idx:
        cgpt.block_zaxpy(
            fineZ.v_obj[i], coarseA.v_obj[0], fineX.v_obj[i], fineY.v_obj[i]
        )
    return fineZ


def normalize(coarse_grid, fine):
    assert type(coarse_grid) == gpt.grid
    coarse_tmp = gpt.complex(coarse_grid)
    zero = gpt.lattice(fine)
    zero[:] = 0.0
    innerProduct(coarse_tmp, fine, fine)
    # TODO: this line is ugly and should probably move to ET
    coarse_tmp[:] = coarse_tmp[:] ** -0.5
    zaxpy(fine, coarse_tmp, fine, zero)
    return fine


def normalize_other(coarse_grid, fine):
    assert type(coarse_grid) == gpt.grid
    coarse_tmp = gpt.complex(coarse_grid)
    zero = gpt.lattice(fine)
    zero[:] = 0.0
    innerProduct_other(coarse_tmp, fine, fine)
    # TODO: this line is ugly and should probably move to ET
    coarse_tmp[:] = coarse_tmp[:] ** -0.5
    zaxpy(fine, coarse_tmp, fine, zero)
    return fine


def orthonormalize_virtual(coarse_grid, basis):
    assert type(coarse_grid) == gpt.grid
    coarse_tmp = gpt.complex(coarse_grid)
    for idx_v, v in enumerate(basis):
        for idx_u, u in enumerate(basis[:idx_v]):
            innerProduct(coarse_tmp, u, v)
            coarse_tmp *= -1
            zaxpy(v, coarse_tmp, u, v)
        normalize(coarse_grid, v)


def orthonormalize_virtual_other(coarse_grid, basis):
    assert type(coarse_grid) == gpt.grid
    coarse_tmp = gpt.complex(coarse_grid)
    for idx_v, v in enumerate(basis):
        for idx_u, u in enumerate(basis[:idx_v]):
            innerProduct_other(coarse_tmp, u, v)
            coarse_tmp *= -1
            zaxpy(v, coarse_tmp, u, v)
        normalize_other(coarse_grid, v)


def orthonormalize(coarse_grid, basis):
    assert type(coarse_grid) == gpt.grid
    assert len(basis[0].v_obj) == 1  # for now
    coarse_tmp = gpt.complex(coarse_grid)
    cgpt.block_orthonormalize(coarse_tmp.v_obj[0], basis)


def check_orthogonality(coarse_grid, basis, tol=None):
    assert type(coarse_grid) == gpt.grid
    nbasis = len(basis)
    iproj, eproj = (
        gpt.vcomplex(coarse_grid, nbasis),
        gpt.vcomplex(coarse_grid, nbasis),
    )
    for i, v in enumerate(basis):
        project(iproj, v, basis)
        eproj[:] = 0.0
        eproj[:, :, :, :, i] = 1.0
        err2 = gpt.norm2(eproj - iproj)
        gpt.message(
            f"DEBUG: Orthogonality for basis vector {i:d}: err2 = {err2:e} tol = {tol:e}"
        )
        if tol is not None:
            assert err2 <= tol
            gpt.message(
                f"Orthogonality check passed for basis vector {i:d}: {err2:e} <= {tol:e}"
            )
        else:
            gpt.message(f"Orthogonality check error for basis vector {i:d}: {err2:e}")
