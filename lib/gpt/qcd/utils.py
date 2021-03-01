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


def reunitize(gauge):
#     'project' site-local matrix to SU(N).
#        * roughly equivalent to Grids 'ProjectOnGroup'
#        * uses the "modified Gram-Schmidt process"
#        * intended to remove numerical rounding errors during HMC
#        * can be unstable for very large N or input far away from SU(N) (not an issue for intended usecase)
    if type(gauge) == list:
        for u in gauge:
            reunitize(u)
        return

    t_total, t_sep, t_merge, t_c, t_norm, t_det = 0, 0, 0, 0, 0, 0

    t_total -= gpt.time()

    assert type(gauge) == gpt.lattice
    shape = gauge.otype.shape
    assert len(shape) == 2 and shape[0] == shape[1]
    n_color = shape[0]

    t_sep -= gpt.time()
    tmp = gpt.separate_color(gauge)
    t_sep += gpt.time()

    c = gpt.complex(tmp[0, 0].grid)
    norm = gpt.complex(tmp[0, 0].grid)

    # step 1: (modified) Gram-Schmidt process to get a unitary matrix
    for i in range(n_color):
        for j in range(i):
            t_c -= gpt.time()
            c @= gpt.conj(tmp[j, 0]) * tmp[i, 0]
            for k in range(1, n_color):
                c += gpt.conj(tmp[j, k]) * tmp[i, k]

            for k in range(n_color):
                tmp[i, k] -= c * tmp[j, k]
            t_c += gpt.time()

        t_norm -= gpt.time()
        norm @= gpt.component.pow(2) * gpt.component.abs(tmp[i, 0])
        for k in range(1, n_color):
            norm += gpt.component.pow(2) * gpt.component.abs(tmp[i, k])
        norm @= gpt.component.inv(gpt.component.sqrt(norm))

        for k in range(n_color):
            tmp[i, k] *= norm
        t_norm += gpt.time()

    t_merge -= gpt.time()
    gpt.merge_color(gauge, tmp)
    t_merge += gpt.time()

    # step 2: fix the determinant (NOTE: Grids 'ProjectOnGroup' skips this step)
    t_det -= gpt.time()
    gauge *= gpt.component.pow(-1. / n_color) * gpt.matrix.det(gauge)
    t_det += gpt.time()

    t_total += gpt.time()

    if gpt.default.is_verbose("reunitize"):
        t_unprofiled = t_total - t_sep - t_merge - t_norm - t_c - t_det
        gpt.message("reunitize: total", t_total, "s")
        gpt.message("reunitize: t_separate", t_sep, "s", round(100 * t_sep / t_total, 1), "%")
        gpt.message("reunitize: t_merge", t_merge, "s", round(100 * t_merge / t_total, 1), "%")
        gpt.message("reunitize: t_c", t_c, "s", round(100 * t_c / t_total, 1), "%")
        gpt.message("reunitize: t_norm", t_norm, "s", round(100 * t_norm / t_total, 1), "%")
        gpt.message("reunitize: t_det", t_det, "s", round(100 * t_det / t_total, 1), "%")
        gpt.message("reunitize: unprofiled", t_unprofiled, "s", round(100 * t_unprofiled / t_total, 1), "%")

