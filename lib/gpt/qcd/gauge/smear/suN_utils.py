#
#    GPT - Grid Python Toolkit
#    Copyright (C) 2020  Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)
#                  2020  Lorenzo Barca    (lorenzo1.barca@ur.de)
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
import numpy as np
import gpt as g
from gpt.params import params_convention

#  Extract an unnormalized SU(2) matrix from a GL(3,C) matrix
#  Project a complex Nc x Nc matrix W onto SU(Nc=3) by maximizing Tr(VW)

#  Should I rename it to "project_to_suN"?
def project_to_su3(u, w, su2_index):
    v = g.eval(u * w)
    grid = u.grid
    Nc = u.otype.Nc
    Vol = u[:].shape[0]
    r = np.empty((Vol, 4), dtype=float)
    su2_extract(r, v, su2_index)

    # now project onto SU(2)
    r_l = np.empty(Vol, dtype=complex)
    r_l = np.sqrt(r[:, 0]**2 + r[:, 1]**2 + r[:, 2]**2 + r[:, 3]**2)

    #  normalize
    #  Fill   (r[0]/r_l, -r[1]/r_l, -r[2]/r_l, -r[3]/r_l)
    a_0 = np.array([1., 0., 0., 0.], dtype=float)
    a = np.empty((Vol, 4), dtype=float)
    a[:, 0] = r[:, 0] / r_l
    a[:, 1] = -r[:, 1] / r_l
    a[:, 2] = -r[:, 2] / r_l
    a[:, 3] = -r[:, 3] / r_l

    suN_fill(v, a, su2_index)
    tmp = g.eval(v * u)

    return tmp

#     * Extract components r_k proportional to SU(2) submatrix su2_index
#     * from the "SU(3)" matrix V. The SU(2) matrix is parameterized in the
#     * sigma matrix basis.
#     * Then compute the b(k) of $ A_SU(2) = b0 I + i sum_k bk sigma_k $

def su2_extract(r, source, su2_index):
    source = g.eval(source)
    Nc = source.otype.Nc
    found, del_i, index = 0, 0, -1
    while(del_i < (Nc-1) and found ==0):
        del_i += 1
        for i1 in range(Nc - del_i):
            index +=1
            if (index == su2_index):
                found = 1
                break
    i2 = i1 + del_i

    r[:, 0, np.newaxis, np.newaxis] = source[:, :, :, :, i1, i1].real + source[:, :, :, :, i2, i2].real
    r[:, 1, np.newaxis, np.newaxis] = source[:, :, :, :, i1, i2].imag + source[:, :, :, :, i2, i1].imag
    r[:, 2, np.newaxis, np.newaxis] = source[:, :, :, :, i1, i2].real - source[:, :, :, :, i2, i1].real
    r[:, 3, np.newaxis, np.newaxis] = source[:, :, :, :, i1, i1].imag - source[:, :, :, :, i2, i2].imag


# Fill a dest(su2_index) <-- r_0, r_1, r_2, r_3  under a subset
#
#   * Fill an SU(Nc) matrix V with the SU(2) submatrix su2_index
#   * parameterized by b_k in the sigma matrix basis.
#   *
#   * Fill in B from B_SU(2) = b0 + i sum_k bk sigma_k
#
def suN_fill(dest, r, su2_index):

#     /* Determine the SU(N) indices corresponding to the SU(2) indices */
#     /* of the SU(2) subgroup $3 */
    Nc = dest.otype.Nc
    found, del_i, index = 0, 0, -1
    while (del_i < (Nc-1) and found == 0):
        del_i +=1
        for i1 in range(Nc-del_i):
            index +=1
            if (index == su2_index):
                found = 1
                break
    i2 = i1 + del_i
    if (found == 0): print("Trouble with SU(2) subgroup index")

#
#     * Insert the b(k) of A_SU(2) = b0 + i sum_k bk sigma_k
#     * back into the SU(N) matrix
#

    dest[:] = 0
    dest[:, :, :, :, 0, 0] = 1
    dest[:, :, :, :, 1, 1] = 1
    dest[:, :, :, :, 2, 2] = 1
    tmp = g.separate_color(dest)

    tmp[i1, i1] = r[:, 0] + 1j * r[:, 3]
    tmp[i1, i2] = r[:, 2] + 1j * r[:, 1]
    tmp[i2, i1] = -r[:, 2] + 1j * r[:, 1]
    tmp[i2, i2] = r[:, 0] - 1j * r[:, 3]

    g.merge_color(dest, tmp)
