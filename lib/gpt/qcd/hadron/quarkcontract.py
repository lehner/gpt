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


def quark_contract_xx(msc1, msc2, grid):
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

    c1 = gpt.separate_color(msc1)
    c2 = gpt.separate_color(msc2)

    y = {
        (0, 0): gpt.eval(c1[1, 1] * c2[2, 2] - c1[1, 2] * c2[2, 1] - c1[2, 1] * c2[1, 2] + c1[2, 2] * c2[1, 1]),
        (0, 1): gpt.eval(c1[2, 1] * c2[0, 2] - c1[2, 2] * c2[0, 1] - c1[0, 1] * c2[2, 2] + c1[0, 2] * c2[2, 1]),
        (0, 2): gpt.eval(c1[0, 1] * c2[1, 2] - c1[0, 2] * c2[1, 1] - c1[1, 1] * c2[0, 2] + c1[1, 2] * c2[0, 1]),
        (1, 0): gpt.eval(c1[1, 2] * c2[2, 0] - c1[1, 0] * c2[2, 2] - c1[2, 2] * c2[1, 0] + c1[2, 0] * c2[1, 2]),
        (1, 1): gpt.eval(c1[2, 2] * c2[0, 0] - c1[2, 0] * c2[0, 2] - c1[0, 2] * c2[2, 0] + c1[0, 0] * c2[2, 2]),
        (1, 2): gpt.eval(c1[0, 2] * c2[1, 0] - c1[0, 0] * c2[1, 2] - c1[1, 2] * c2[0, 0] + c1[1, 0] * c2[0, 2]),
        (2, 0): gpt.eval(c1[1, 0] * c2[2, 1] - c1[1, 1] * c2[2, 0] - c1[2, 0] * c2[1, 1] + c1[2, 1] * c2[1, 0]),
        (2, 1): gpt.eval(c1[2, 0] * c2[0, 1] - c1[2, 1] * c2[0, 0] - c1[0, 0] * c2[2, 1] + c1[0, 1] * c2[2, 0]),
        (2, 2): gpt.eval(c1[0, 0] * c2[1, 1] - c1[0, 1] * c2[1, 0] - c1[1, 0] * c2[0, 1] + c1[1, 1] * c2[0, 0])
    }

    dst = gpt.mcolor(grid)
    gpt.merge_color(dst, y)

    return dst


def quark_contract_12(mspincolor1, mspincolor2):
    grid = mspincolor1.grid
    dst = gpt.mspincolor(grid)
    dst[:] = 0
    mspin = gpt.separate_spin(dst)
    mspin1 = gpt.separate_spin(mspincolor1)
    mspin2 = gpt.separate_spin(mspincolor2)

    for s1 in range(4):
        for s2 in range(4):
            for k in range(4):
                mspin[s1, s2] += quark_contract_xx(mspin1[k, k], mspin2[s1, s2], grid)

    gpt.merge_spin(dst, mspin)
    return dst


def quark_contract_13(mspincolor1, mspincolor2):
    grid = mspincolor1.grid
    dst = gpt.mspincolor(grid)
    dst[:] = 0
    mspin = gpt.separate_spin(dst)
    mspin1 = gpt.separate_spin(mspincolor1)
    mspin2 = gpt.separate_spin(mspincolor2)

    for s1 in range(4):
        for s2 in range(4):
            for k in range(4):
                mspin[s1, s2] += quark_contract_xx(mspin1[k, s1], mspin2[k, s2], grid)

    gpt.merge_spin(dst, mspin)
    return dst


def quark_contract_14(mspincolor1, mspincolor2):
    grid = mspincolor1.grid
    dst = gpt.mspincolor(grid)
    dst[:] = 0
    mspin = gpt.separate_spin(dst)
    mspin1 = gpt.separate_spin(mspincolor1)
    mspin2 = gpt.separate_spin(mspincolor2)

    for s1 in range(4):
        for s2 in range(4):
            for k in range(4):
                mspin[s1, s2] += quark_contract_xx(mspin1[k, s1], mspin2[s2, k], grid)

    gpt.merge_spin(dst, mspin)
    return dst


def quark_contract_23(mspincolor1, mspincolor2):
    grid = mspincolor1.grid
    dst = gpt.mspincolor(grid)
    dst[:] = 0
    mspin = gpt.separate_spin(dst)
    mspin1 = gpt.separate_spin(mspincolor1)
    mspin2 = gpt.separate_spin(mspincolor2)

    for s1 in range(4):
        for s2 in range(4):
            for k in range(4):
                mspin[s1, s2] += quark_contract_xx(mspin1[s1, k], mspin2[k, s2], grid)

    gpt.merge_spin(dst, mspin)
    return dst


def quark_contract_24(mspincolor1, mspincolor2):
    grid = mspincolor1.grid
    dst = gpt.mspincolor(grid)
    dst[:] = 0
    mspin = gpt.separate_spin(dst)
    mspin1 = gpt.separate_spin(mspincolor1)
    mspin2 = gpt.separate_spin(mspincolor2)

    for s1 in range(4):
        for s2 in range(4):
            for k in range(4):
                mspin[s1, s2] += quark_contract_xx(mspin1[s1, k], mspin2[s2, k], grid)

    gpt.merge_spin(dst, mspin)
    return dst


def quark_contract_34(mspincolor1, mspincolor2):
    grid = mspincolor1.grid
    dst = gpt.mspincolor(grid)
    dst[:] = 0
    mspin = gpt.separate_spin(dst)
    mspin1 = gpt.separate_spin(mspincolor1)
    mspin2 = gpt.separate_spin(mspincolor2)

    for s1 in range(4):
        for s2 in range(4):
            for k in range(4):
                mspin[s1, s2] += quark_contract_xx(mspin1[s1, s2], mspin2[k, k], grid)

    gpt.merge_spin(dst, mspin)
    return dst
