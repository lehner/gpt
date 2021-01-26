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

    dst = gpt.mspincolor(mspincolor1.grid)
    dst[:] = 0
    sdst = gpt.separate_spin(dst)

    mcolor1 = gpt.separate_spin(mspincolor1)
    mcolor2 = gpt.separate_spin(mspincolor2)

    comps1, comps2, target = dict(), dict(), dict()
    for key in components.keys():

        assert key not in target, f"Found duplicate key: {key}"
        target[key] = gpt.separate_color(sdst[key])

        for comps in components[key]:
            c0_0, c0_1, c1_0, c1_1 = comps
            if (c0_0, c0_1) not in comps1:
                comps1[(c0_0, c0_1)] = gpt.separate_color(mcolor1[(c0_0, c0_1)])
            if (c1_0, c1_1) not in comps2:
                comps2[(c1_0, c1_1)] = gpt.separate_color(mcolor2[(c1_0, c1_1)])

    for key in components.keys():
        tmp = target[key]

        for comps in components[key]:
            c0_0, c0_1, c1_0, c1_1 = comps

            c1 = comps1[(c0_0, c0_1)]
            c2 = comps2[(c1_0, c1_1)]

            tmp[(0, 0)] += gpt.eval(c1[1, 1] * c2[2, 2] - c1[1, 2] * c2[2, 1] - c1[2, 1] * c2[1, 2] + c1[2, 2] * c2[1, 1])
            tmp[(0, 1)] += gpt.eval(c1[2, 1] * c2[0, 2] - c1[2, 2] * c2[0, 1] - c1[0, 1] * c2[2, 2] + c1[0, 2] * c2[2, 1])
            tmp[(0, 2)] += gpt.eval(c1[0, 1] * c2[1, 2] - c1[0, 2] * c2[1, 1] - c1[1, 1] * c2[0, 2] + c1[1, 2] * c2[0, 1])
            tmp[(1, 0)] += gpt.eval(c1[1, 2] * c2[2, 0] - c1[1, 0] * c2[2, 2] - c1[2, 2] * c2[1, 0] + c1[2, 0] * c2[1, 2])
            tmp[(1, 1)] += gpt.eval(c1[2, 2] * c2[0, 0] - c1[2, 0] * c2[0, 2] - c1[0, 2] * c2[2, 0] + c1[0, 0] * c2[2, 2])
            tmp[(1, 2)] += gpt.eval(c1[0, 2] * c2[1, 0] - c1[0, 0] * c2[1, 2] - c1[1, 2] * c2[0, 0] + c1[1, 0] * c2[0, 2])
            tmp[(2, 0)] += gpt.eval(c1[1, 0] * c2[2, 1] - c1[1, 1] * c2[2, 0] - c1[2, 0] * c2[1, 1] + c1[2, 1] * c2[1, 0])
            tmp[(2, 1)] += gpt.eval(c1[2, 0] * c2[0, 1] - c1[2, 1] * c2[0, 0] - c1[0, 0] * c2[2, 1] + c1[0, 1] * c2[2, 0])
            tmp[(2, 2)] += gpt.eval(c1[0, 0] * c2[1, 1] - c1[0, 1] * c2[1, 0] - c1[1, 0] * c2[0, 1] + c1[1, 1] * c2[0, 0])

    for key in components.keys():
        gpt.merge_color(sdst[key], target[key])
    gpt.merge_spin(dst, sdst)

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
