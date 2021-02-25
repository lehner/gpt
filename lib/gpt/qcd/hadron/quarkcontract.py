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

import gpt
import numpy as np


def quark_contract_xx(mspincolor1, mspincolor2, components):
    """
    This routine is written for Nc = 3

    y^{k2, k1} = \\sum_{i1, i2, j1, j2} \\epsilon^{i1, j1, k1} \\epsilon^{i2, j2, k2} xc1^{i1, i2} xc2^{j1, j2}
    Permutations: +(0, 1, 2), +(1, 2, 0), +(2, 0, 1),
                     -(1, 0, 2), -(0, 2, 1), -(2, 1, 0)
    i.e.
    - y^{0, 0} = \\epsilon^{i1, j1, 0} \\epsilon^{i2, j2, 0} xc1^{i1, i2} xc2^{j1, j2}
        Permutations: +(1, 2, 0), -(2, 1, 0);         +(1, 2, 0), -(2, 1, 0)
    - y^{0, 1} = \\epsilon^{i1, j1, 1} \\epsilon^{i2, j2, 0} xc1^{i1, i2} xc2^{j1, j2}
        Permutations: +(2, 0, 1), -(0, 2, 1);         +(1, 2, 0), -(2, 1, 0)
    - y^{0, 2} = \\epsilon^{i1, j1, 2} \\epsilon^{i2, j2, 0} xc1^{i1, i2} xc2^{j1, j2}
        Permutations: +(0, 1, 2), -(1, 0, 2)          +(1, 2, 0), -(2, 1, 0)
    - y^{1, 0} = \\epsilon^{i1, j1, 0} \\epsilon^{i2, j2, 1} xc1^{i1, i2} xc2^{j1, j2}
        Permutations: +(1, 2, 0), -(2, 1, 0)          +(2, 0, 1), -(0, 2, 1)
    - y^{1, 1} = \\epsilon^{i1, j1, 1} \\epsilon^{i2, j2, 1} xc1^{i1, i2} xc2^{j1, j2}
        Permutations: +(2, 0, 1), -(0, 2, 1)          +(2, 0, 1), -(0, 2, 1)
    - y^{1, 2} = \\epsilon^{i1, j1, 2} \\epsilon^{i2, j2, 1} xc1^{i1, i2} xc2^{j1, j2}
        Permutations: +(0, 1, 2), -(1, 0, 2)          +(2, 0, 1), -(0, 2, 1)
    - y^{2, 0} = \\epsilon^{i1, j1, 0} \\epsilon^{i2, j2, 2} xc1^{i1, i2} xc2^{j1, j2}
        Permutations: +(1, 2, 0), -(2, 1, 0)          +(0, 1, 2), -(1, 0, 2)
    - y^{2, 1} = \\epsilon^{i1, j1, 1} \\epsilon^{i2, j2, 2} xc1^{i1, i2} xc2^{j1, j2}
        Permutations: +(2, 0, 1), -(0, 2, 1)          +(0, 1, 2), -(1, 0, 2)
    - y^{2, 2} = \\epsilon^{i1, j1, 2} \\epsilon^{i2, j2, 2} xc1^{i1, i2} xc2^{j1, j2}
        Permutations: +(0, 1, 2), -(1, 0, 2)          +(0, 1, 2), -(1, 0, 2)
    """

    t_separatespin, t_separatecolor, t_create, t_bilinear, t_merge = 0.0, 0.0, 0.0, 0.0, 0.0
    t_start = gpt.time()

    grid = mspincolor1.grid

    t_separatespin -= gpt.time()
    mcolor1 = gpt.separate_spin(mspincolor1)
    mcolor2 = gpt.separate_spin(mspincolor2)
    t_separatespin += gpt.time()

    t_create -= gpt.time()
    dst = gpt.mspincolor(grid)
    spinsep_dst = {(ii // 4, ii % 4): gpt.mcolor(grid) for ii in range(16)}
    bilinear_result = [gpt.complex(grid) for _ in range(9)]
    t_create += gpt.time()

    leftbase = np.array([
        [4, 5, 7, 8], [7, 8, 1, 2], [1, 2, 4, 5],
        [5, 3, 8, 6], [8, 6, 2, 0], [2, 0, 5, 3],
        [3, 4, 6, 7], [6, 7, 0, 1], [0, 1, 3, 4]
    ], dtype=np.int32)
    rightbase = np.array([
        [8, 7, 5, 4], [2, 1, 8, 7], [5, 4, 2, 1],
        [6, 8, 3, 5], [0, 2, 6, 8], [3, 5, 0, 2],
        [7, 6, 4, 3], [1, 0, 7, 6], [4, 3, 1, 0]
    ], dtype=np.int32)

    comps1, comps2, used_keys = dict(), dict(), []
    for spin_key in components.keys():
        assert spin_key not in used_keys, f"Found duplicate key: {key}"
        used_keys.append(spin_key)

        lefts, rights = [], []
        bilin_coeffs, bilin_leftbasis, bilin_rightbasis = [], [], []
        for nn, comps in enumerate(components[spin_key]):
            c0_0, c0_1, c1_0, c1_1 = comps

            if (c0_0, c0_1) not in comps1:
                t_separatecolor -= gpt.time()
                comps1[c0_0, c0_1] = gpt.separate_color(mcolor1[c0_0, c0_1])
                t_separatecolor += gpt.time()
            if (c1_0, c1_1) not in comps2:
                t_separatecolor -= gpt.time()
                comps2[c1_0, c1_1] = gpt.separate_color(mcolor2[c1_0, c1_1])
                t_separatecolor += gpt.time()

            for ii in range(9):
                lefts.append(comps1[c0_0, c0_1][ii // 3, ii % 3])
                rights.append(comps2[c1_0, c1_1][ii // 3, ii % 3])

            bilin_coeffs = np.append(bilin_coeffs, [1.0, -1.0, -1.0, +1.0])
            bilin_leftbasis.append(leftbase + nn * 9)
            bilin_rightbasis.append(rightbase + nn * 9)

        bilin_coeffs = np.array([bilin_coeffs for _ in range(9)], dtype=np.int32)
        bilin_leftbasis = np.concatenate(bilin_leftbasis, axis=1)
        bilin_rightbasis = np.concatenate(bilin_rightbasis, axis=1)

        t_bilinear -= gpt.time()
        gpt.bilinear_combination(
            bilinear_result,
            lefts,
            rights,
            bilin_coeffs,
            bilin_leftbasis,
            bilin_rightbasis
        )
        t_bilinear += gpt.time()

        cmat_dict = dict()
        for ii in range(9):
            cmat_dict[ii // 3, ii % 3] = bilinear_result[ii]

        t_merge -= gpt.time()
        gpt.merge_color(spinsep_dst[spin_key], cmat_dict)
        t_merge += gpt.time()

    t_merge -= gpt.time()
    gpt.merge_spin(dst, spinsep_dst)
    t_merge += gpt.time()

    t_total = gpt.time() - t_start
    t_profiled = t_separatespin + t_separatecolor + t_create + t_bilinear + t_merge

    if gpt.default.is_verbose("quark_contract_xx"):
        gpt.message("quark_contract_xx: total", t_total, "s")
        gpt.message("quark_contract_xx: t_separatespin", t_separatespin, "s", round(100 * t_separatespin / t_total, 1), "%")
        gpt.message("quark_contract_xx: t_separatecolor", t_separatecolor, "s", round(100 * t_separatecolor / t_total, 1), "%")
        gpt.message("quark_contract_xx: t_create", t_create, "s", round(100 * t_create / t_total, 1), "%")
        gpt.message("quark_contract_xx: t_bilinear", t_bilinear, "s", round(100 * t_bilinear / t_total, 1), "%")
        gpt.message("quark_contract_xx: t_merge", t_merge, "s", round(100 * t_merge / t_total, 1), "%")
        gpt.message("quark_contract_xx: unprofiled", t_total - t_profiled, "s", round(100 * (t_total - t_profiled) / t_total, 1), "%")
    return dst


def quark_contract_12(mspincolor1, mspincolor2):
    components = dict()
    for s1 in range(4):
        for s2 in range(4):
            components[(s1, s2)] = []
            for k in range(4):
                components[(s1, s2)].append([k, k, s1, s2])
    return quark_contract_xx(mspincolor1, mspincolor2, components)


def quark_contract_13(mspincolor1, mspincolor2):
    components = dict()
    for s1 in range(4):
        for s2 in range(4):
            components[(s1, s2)] = []
            for k in range(4):
                components[(s1, s2)].append([k, s1, k, s2])
    return quark_contract_xx(mspincolor1, mspincolor2, components)


def quark_contract_14(mspincolor1, mspincolor2):
    components = dict()
    for s1 in range(4):
        for s2 in range(4):
            for k in range(4):
                components[(s1, s2)].append([k, s1, s2, k])
    return quark_contract_xx(mspincolor1, mspincolor2, components)


def quark_contract_23(mspincolor1, mspincolor2):
    components = dict()
    for s1 in range(4):
        for s2 in range(4):
            components[(s1, s2)] = []
            for k in range(4):
                components[(s1, s2)].append([s1, k, k, s2])
    return quark_contract_xx(mspincolor1, mspincolor2, components)


def quark_contract_24(mspincolor1, mspincolor2):
    components = dict()
    for s1 in range(4):
        for s2 in range(4):
            components[(s1, s2)] = []
            for k in range(4):
                components[(s1, s2)].append([s1, k, s2, k])
    return quark_contract_xx(mspincolor1, mspincolor2, components)


def quark_contract_34(mspincolor1, mspincolor2):
    components = dict()
    for s1 in range(4):
        for s2 in range(4):
            components[(s1, s2)] = []
            for k in range(4):
                components[(s1, s2)].append([s1, s2, k, k])
    return quark_contract_xx(mspincolor1, mspincolor2, components)
