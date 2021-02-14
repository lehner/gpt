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
import gpt, numpy


def gamma5(src):
    if hasattr(src.otype, "fundamental"):
        nbasis = src.otype.shape[0]
        assert nbasis % 2 == 0
        nb = nbasis // 2
        return gpt.vcomplex([1] * nb + [-1] * nb, nbasis)
    else:
        return gpt.gamma[5]


def split_chiral(basis, factor=None):
    nbasis = len(basis)
    assert nbasis % 2 == 0
    nb = nbasis // 2
    factor = 0.5 if factor is None else factor
    g5 = gamma5(basis[0])
    tmp = gpt.lattice(basis[0])
    for n in range(nb):
        tmp @= g5 * basis[n]
        basis[n + nb] @= (basis[n] - tmp) * factor
        basis[n] @= (basis[n] + tmp) * factor


def unsplit_chiral(basis, factor=None):
    nbasis = len(basis)
    assert nbasis % 2 == 0
    nb = nbasis // 2
    factor = 0.5 if factor is None else factor
    rev_factor = 0.5 / factor
    for n in range(nb):
        basis[n] @= (basis[n] + basis[n + nb]) * rev_factor


def prefactor_dagger(A, v_idx=None):
    assert type(A) == gpt.lattice
    nbasis = A.otype.shape[0]
    assert nbasis % 2 == 0
    nb = nbasis // 2

    factor = numpy.ones((nbasis, nbasis), dtype=gpt.double.real_dtype)
    factor[0:nb, nb:nbasis] *= -1.0  # upper right block
    factor[nb:nbasis, 0:nb] *= -1.0  # lower left block

    if v_idx is None:
        return factor

    # extract the subblock of the prefactor belonging to v_idx
    assert v_idx < len(A.v_obj)
    nrow_v_idx = int(len(A.v_obj) ** 0.5)
    nbasis_v_idx = A.otype.v_n1[0]
    row_v_idx = v_idx % nrow_v_idx
    col_v_idx = v_idx // nrow_v_idx

    return (
        factor[
            row_v_idx * nbasis_v_idx : (row_v_idx + 1) * nbasis_v_idx,
            col_v_idx * nbasis_v_idx : (col_v_idx + 1) * nbasis_v_idx,
        ]
        .reshape(nbasis_v_idx * nbasis_v_idx)
        .tolist()
    )


def nearest_neighbor_operator(fine_matrix, coarse_grid, basis, params):
    A = [gpt.mcomplex(coarse_grid, len(basis)) for i in range(9)]

    gpt.coarse.create_links(
        A, fine_matrix, basis, make_hermitian=params["make_hermitian"], save_links=True
    )

    level = 1 if isinstance(fine_matrix.otype[0], gpt.ot_matrix_singlet) else 0
    return gpt.qcd.fermion.coarse_fermion(A, level=level)
