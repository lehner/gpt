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
def project_onto_suN(dest, u_unprojected, params):
    t_total = - gpt.time()
    t_trace, t_projectstep = 0.0, 0.0

    vol = dest.grid.fsites
    t_trace -= gpt.time()
    old_trace = gpt.sum(gpt.trace(dest * u_unprojected)).real / (vol * 3)
    t_trace += gpt.time()

    for _ in range(params["max_iteration"]):
        # perform a single projection step
        t_projectstep -= gpt.time()
        project_to_suN_step(dest, u_unprojected)
        t_projectstep += gpt.time()

        # calculate new trace
        t_trace -= gpt.time()
        new_trace = gpt.sum(gpt.trace(dest * u_unprojected)).real / (vol * 3)
        t_trace += gpt.time()

        epsilon = np.abs((new_trace - old_trace) / old_trace)
        gpt.message(f"APE iter {_}, epsilon: {epsilon}")

        if epsilon < params["accuracy"]:
            break
        old_trace = new_trace
    else:
        raise RuntimeError("Projection to SU(3) did not converge.")

    t_total += gpt.time()

    if gpt.default.is_verbose("project_onto_suN"):
        t_profiled = t_trace + t_projectstep
        t_unprofiled = t_total - t_profiled

        gpt.message("project_onto_suN: total", t_total, "s")
        gpt.message("project_onto_suN: t_trace", t_trace, "s", round(100 * t_trace / t_total, 1), "%")
        gpt.message("project_onto_suN: t_projectstep", t_projectstep, "s", round(100 * t_projectstep / t_total, 1), "%")
        gpt.message("project_onto_suN: unprofiled", t_unprofiled, "s", round(100 * t_unprofiled / t_total, 1), "%")


def project_to_suN_step(dest, unprojected):
    t_total = - gpt.time()
    t_product, t_separate, t_merge, t_su2extract, t_su2fill, t_calcnorm, t_applynorm = [0.0 for _ in range(7)]

    vol = dest.grid.fsites
    n_colors = dest.otype.Nc
    tmp = gpt.mcolor(dest.grid)

    zero = gpt.complex(dest.grid)
    zero[:] = 0.0
    one = gpt.complex(dest.grid)
    one[:] = 1.0

    square = gpt.component.pow(2)
    norm = gpt.complex(dest.grid)

    for su2_index in range(n_colors * (n_colors - 1) // 2):

        index, i1, i2 = 0, None, None
        for ii in range(1, n_colors):
            for jj in range(n_colors - ii):
                if index == su2_index and i1 is None:
                    i1 = jj
                    i2 = ii + jj
                index += 1

        t_product -= gpt.time()
        tmp @= dest * unprojected
        t_product += gpt.time()

        t_separate -= gpt.time()
        tmp_sep = gpt.separate_color(tmp)
        t_separate += gpt.time()

        t_su2extract -= gpt.time()
        su2_components = extract_su2_components(tmp_sep, [i1, i2])
        t_su2extract += gpt.time()

        t_calcnorm -= gpt.time()
        norm @= gpt.component.inv(gpt.component.sqrt(gpt.eval(
            su2_components[0] * su2_components[0] +
            su2_components[1] * su2_components[1] +
            su2_components[2] * su2_components[2] +
            su2_components[3] * su2_components[3]
        )))
        t_calcnorm += gpt.time()

        t_applynorm -= gpt.time()
        su2_components[0] @= su2_components[0] * norm
        su2_components[1] @= - su2_components[1] * norm
        su2_components[2] @= - su2_components[2] * norm
        su2_components[3] @= - su2_components[3] * norm
        t_applynorm += gpt.time()

        t_su2fill -= gpt.time()
        fill_su2_components_into_suN(tmp_sep, su2_components, [i1, i2], cache=[zero, one])
        t_su2fill += gpt.time()

        t_merge -= gpt.time()
        gpt.merge_color(tmp, tmp_sep)
        t_merge += gpt.time()

        t_product -= gpt.time()
        dest @= tmp * dest
        t_product += gpt.time()

    t_total += gpt.time()

    if gpt.default.is_verbose("project_to_suN_step"):
        t_profiled = t_product + t_separate + t_merge + t_su2extract + t_su2fill + t_calcnorm + t_applynorm
        t_unprofiled = t_total - t_profiled

        gpt.message("project_to_suN_step: total", t_total, "s")
        gpt.message("project_to_suN_step: t_product", t_product, "s", round(100 * t_product / t_total, 1), "%")
        gpt.message("project_to_suN_step: t_separate", t_separate, "s", round(100 * t_separate / t_total, 1), "%")
        gpt.message("project_to_suN_step: t_merge", t_merge, "s", round(100 * t_merge / t_total, 1), "%")
        gpt.message("project_to_suN_step: t_su2extract", t_su2extract, "s", round(100 * t_su2extract / t_total, 1), "%")
        gpt.message("project_to_suN_step: t_su2fill", t_su2fill, "s", round(100 * t_su2fill / t_total, 1), "%")
        gpt.message("project_to_suN_step: t_calcnorm", t_calcnorm, "s", round(100 * t_calcnorm / t_total, 1), "%")
        gpt.message("project_to_suN_step: t_applynorm", t_applynorm, "s", round(100 * t_applynorm / t_total, 1), "%")
        gpt.message("project_to_suN_step: unprofiled", t_unprofiled, "s", round(100 * t_unprofiled / t_total, 1), "%")


#     * Extract components r_k proportional to SU(2) submatrix su2_index
#     * from the "SU(3)" matrix V. The SU(2) matrix is parameterized in the
#     * sigma matrix basis.
#     * Then compute the b(k) of $ A_SU(2) = b0 I + i sum_k bk sigma_k $
def extract_su2_components(suN_matrix, su2_indices):
    if isinstance(suN_matrix, gpt.lattice):
        separated = gpt.separate_color(suN_matrix)
    elif isinstance(suN_matrix, dict):
        n_keys = len(suN_matrix)
        n_colors = int(np.sqrt(n_keys))
        assert (n_colors - 1, n_colors - 1) in suN_matrix
        separated = suN_matrix

    i1, i2 = su2_indices

    su2_components = [gpt.complex(separated[0, 0].grid) for _ in range(4)]
    su2_components[0] @= gpt.component.real(gpt.eval(separated[i1, i1] + separated[i2, i2]))
    su2_components[1] @= gpt.component.imag(gpt.eval(separated[i1, i2] + separated[i2, i1]))
    su2_components[2] @= gpt.component.real(gpt.eval(separated[i1, i2] - separated[i2, i1]))
    su2_components[3] @= gpt.component.imag(gpt.eval(separated[i1, i1] - separated[i2, i2]))

    return su2_components


#
#   * Fill in B from B_SU(2) = b0 + i sum_k bk sigma_k
#
def fill_su2_components_into_suN(dst, su2_comps, su2_indices, cache=None):

    if isinstance(dst, gpt.lattice):
        n_colors = dst.otype.Nc
        separated = gpt.separate_color(dst)
    elif isinstance(dst, dict):
        n_keys = len(dst)
        n_colors = int(np.sqrt(n_keys))
        assert (int(n_colors) - 1, int(n_colors) - 1) in dst
        separated = dst

    if cache is None:
        zero = gpt.complex(separated[0, 0].grid)
        zero[:] = 0.0
        one = gpt.complex(separated[0, 0].grid)
        one[:] = 1.0
    else:
        zero, one = cache

    i1, i2 = su2_indices

    separated[i1, i1] @= su2_comps[0] + 1j * su2_comps[3]
    separated[i1, i2] @= su2_comps[2] + 1j * su2_comps[1]
    separated[i2, i1] @= - su2_comps[2] + 1j * su2_comps[1]
    separated[i2, i2] @= su2_comps[0] - 1j * su2_comps[3]
    for ii in range(n_colors):
        for jj in range(n_colors):
            if ii not in [i1, i2] or jj not in [i1, i2]:
                separated[ii, jj] @= one if ii == jj else zero

    if isinstance(dst, gpt.lattice):
        gpt.merge_color(dst, separated)
