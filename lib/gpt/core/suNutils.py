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
import gpt
from gpt.params import params_convention

#  Extract an unnormalized SU(2) matrix from a GL(Nc,C) matrix
#  Project a complex Nc x Nc matrix U_mu onto SU(Nc) by maximizing Tr(U_mu_smear * U_unproj)

@params_convention(max_iteration=20, accuracy=1e-20)
def project_onto_suN(dest, u_unprojected, params):  # Need more sanity check and comments
    vol = dest.grid.fsites
    old_trace = np.sum(gpt.slice(gpt.trace(dest * u_unprojected), 3)).real / (vol * 3)

    for _ in range(params["max_iteration"]):
        # perform a single projection step
        project_to_suN_step(dest, u_unprojected)

        # calculate new trace
        new_trace = np.sum(gpt.slice(gpt.trace(dest * u_unprojected), 3)).real / (vol * 3)
        epsilon = np.abs((new_trace - old_trace) / old_trace)
        gpt.message(f"APE iter {_}, epsilon: {epsilon}")

        if epsilon < params["accuracy"]:
            break
        old_trace = new_trace
    else:
        raise RuntimeError("Projection to SU(3) did not converge.")


def project_to_suN_step(dest, unprojected):
    vol = dest.grid.fsites
    n_colors = dest.otype.Nc
    tmp = gpt.mcolor(dest.grid)

    for su2_index in range(n_colors * (n_colors - 1) // 2):
        tmp @= gpt.eval(dest * unprojected)
        su2_comps = extract_normalized_su2_components(tmp, su2_index)

        fill_su2_components_into_suN(tmp, su2_comps, su2_index)
        dest @= gpt.eval(tmp * dest)


#     * Extract components r_k proportional to SU(2) submatrix su2_index
#     * from the "SU(3)" matrix V. The SU(2) matrix is parameterized in the
#     * sigma matrix basis.
#     * Then compute the b(k) of $ A_SU(2) = b0 I + i sum_k bk sigma_k $
def extract_normalized_su2_components(suN_matrix, su2_index):
    suN_matrix = gpt.eval(suN_matrix)
    n_colors = suN_matrix.otype.Nc
    su2_components = [gpt.complex(suN_matrix.grid) for _ in range(4)]

    index, i1, i2 = 0, None, None
    for ii in range(1, n_colors):
        for jj in range(n_colors - ii):
            if index == su2_index and i1 is None:
                i1 = jj
                i2 = ii + jj
            index += 1

    tmp_src = gpt.separate_color(suN_matrix)

    # should be replaced by real and imag functions at some point
    su2_components[0] @= gpt.component.real(gpt.eval(tmp_src[i1, i1] + tmp_src[i2, i2]))
    su2_components[1] @= gpt.component.imag(gpt.eval(tmp_src[i1, i2] + tmp_src[i2, i1]))
    su2_components[2] @= gpt.component.real(gpt.eval(tmp_src[i1, i2] - tmp_src[i2, i1]))
    su2_components[3] @= gpt.component.imag(gpt.eval(tmp_src[i1, i1] - tmp_src[i2, i2]))

    square = gpt.component.pow(2)
    norm = gpt.eval(
        square * su2_components[0] +
        square * su2_components[1] +
        square * su2_components[2] +
        square * su2_components[3]
    )
    norm @= gpt.component.inv(gpt.component.sqrt(norm))

    su2_components[0] @= su2_components[0] * norm
    su2_components[1] @= - su2_components[1] * norm
    su2_components[2] @= - su2_components[2] * norm
    su2_components[3] @= - su2_components[3] * norm

    return su2_components


#
#   * Fill in B from B_SU(2) = b0 + i sum_k bk sigma_k
#
def fill_su2_components_into_suN(dst, su2_comps, su2_index):
    n_colors = dst.otype.Nc
    index, i1, i2 = 0, None, None
    for ii in range(1, n_colors):
        for jj in range(n_colors - ii):
            if index == su2_index and i1 is None:
                i1 = jj
                i2 = ii + jj
            index += 1

    tmp = gpt.separate_color(dst)
    tmp[i1, i1] @= su2_comps[0] + 1j * su2_comps[3]
    tmp[i1, i2] @= su2_comps[2] + 1j * su2_comps[1]
    tmp[i2, i1] @= - su2_comps[2] + 1j * su2_comps[1]
    tmp[i2, i2] @= su2_comps[0] - 1j * su2_comps[3]
    for ii in range(n_colors):
        for jj in range(n_colors):
            if ii not in [i1, i2] or jj not in [i1, i2]:
                tmp[ii, jj][:] = 1.0 if ii == jj else 0.0

    gpt.merge_color(dst, tmp)
