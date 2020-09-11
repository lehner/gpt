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
import cgpt
import gpt
import numpy as np

def ferm_to_prop(p, f, s, c):
    assert len(f.v_obj) == 1 and len(p.v_obj) == 1
    return cgpt.util_ferm2prop(f.v_obj[0], p.v_obj[0], s, c, True)

def prop_to_ferm(f, p, s, c):
    assert len(f.v_obj) == 1 and len(p.v_obj) == 1
    return cgpt.util_ferm2prop(f.v_obj[0], p.v_obj[0], s, c, False)

def reunitize(U):
    """
    'project' site-local matrix to SU(N).
       * roughly equivalent to Grids 'ProjectOnGroup'
       * uses the "modified Gram-Schmidt process"
       * intended to remove numerical rounding errors during HMC
       * can be unstable for very large N or input far away from SU(N) (not an issue for intended usecase)
    """
    if type(U) == list:
        for a in U:
            reunitize(a)
        return
    assert type(U) == gpt.lattice
    shape = U.otype.shape
    assert len(shape) == 2 and shape[0] == shape[1]
    N = shape[0]  # number of colors

    # step 1: (modified) Gram-Schmidt process to get a unitary matrix
    tmp = U[:]
    for i in range(N):
        for j in range(i):
            c = np.einsum("ij,ij->i", np.conj(tmp[:, j, :]), tmp[:, i, :])
            tmp[:, i, :] -= c[:, np.newaxis] * tmp[:, j, :]
        tmp[:, i, :] /= np.linalg.norm(tmp[:, i, :], axis=1, keepdims=True)
    U[:] = tmp[:]

    # step 2: fix the determinant (NOTE: Grids 'ProjectOnGroup' skips this step)
    D = gpt.matrix.det(U)
    D[:] **= -1. / N
    U[:] *= D[:][:, np.newaxis]
