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
    normalized_su2_comps = np.empty((vol, 4, 1, 1), dtype=np.float)

    for su2_index in range(n_colors * (n_colors - 1) // 2):
        tmp @= gpt.eval(dest * unprojected)
        su2_comps = extract_su2_components(tmp, su2_index)

        #  normalize
        #  Fill   (r[0]/r_l, -r[1]/r_l, -r[2]/r_l, -r[3]/r_l)
        norm = np.sqrt(su2_comps[:, 0]**2 + su2_comps[:, 1]**2 + su2_comps[:, 2]**2 + su2_comps[:, 3]**2)

        normalized_su2_comps[:, 0] = su2_comps[:, 0] / norm
        normalized_su2_comps[:, 1] = -su2_comps[:, 1] / norm
        normalized_su2_comps[:, 2] = -su2_comps[:, 2] / norm
        normalized_su2_comps[:, 3] = -su2_comps[:, 3] / norm

        tmp[:] = 0
        for ii in range(n_colors):
            tmp[:, :, :, :, ii, ii] = 1

        fill_su2_components_into_suN(tmp, normalized_su2_comps, su2_index)
        dest @= gpt.eval(tmp * dest)


#     * Extract components r_k proportional to SU(2) submatrix su2_index
#     * from the "SU(3)" matrix V. The SU(2) matrix is parameterized in the
#     * sigma matrix basis.
#     * Then compute the b(k) of $ A_SU(2) = b0 I + i sum_k bk sigma_k $
def extract_su2_components(suN_matrix, su2_index):
    suN_matrix = gpt.eval(suN_matrix)
    n_colors = suN_matrix.otype.Nc
    su2_components = np.empty((suN_matrix.grid.fsites, 4, 1, 1), dtype=np.float)

    index, i1, i2 = 0, None, None
    for ii in range(1, n_colors):
        for jj in range(n_colors - ii):
            if index == su2_index and i1 is None:
                i1 = jj
                i2 = ii + jj
            index += 1

    su2_components[:, 0, :, :] = suN_matrix[:, :, :, :, i1, i1].real + suN_matrix[:, :, :, :, i2, i2].real
    su2_components[:, 1, :, :] = suN_matrix[:, :, :, :, i1, i2].imag + suN_matrix[:, :, :, :, i2, i1].imag
    su2_components[:, 2, :, :] = suN_matrix[:, :, :, :, i1, i2].real - suN_matrix[:, :, :, :, i2, i1].real
    su2_components[:, 3, :, :] = suN_matrix[:, :, :, :, i1, i1].imag - suN_matrix[:, :, :, :, i2, i2].imag

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
    tmp[i1, i1][:] = su2_comps[:, 0] + 1j * su2_comps[:, 3]
    tmp[i1, i2][:] = su2_comps[:, 2] + 1j * su2_comps[:, 1]
    tmp[i2, i1][:] = - su2_comps[:, 2] + 1j * su2_comps[:, 1]
    tmp[i2, i2][:] = su2_comps[:, 0] - 1j * su2_comps[:, 3]

    gpt.merge_color(dst, tmp)
