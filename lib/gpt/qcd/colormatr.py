#
#    GPT - Grid Python Toolkit
#    Copyright (C) 2020  Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)
#                  2020  Lorenzo Barca
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

import gpt as g

def quarkContractXX(msc1 , msc2, grid):

    #
    #   This routine is completely written for Nc = 3
    #
    #   y^{k2, k1} = \sum_{i1, i2, j1, j2} \epsilon^{i1, j1, k1} \epsilon^{i2, j2, k2} xc1^{i1, i2} xc2^{j1, j2}
    #
    #   Permutations: +(0, 1, 2), +(1, 2, 0), +(2, 0, 1),
    #                 -(1, 0, 2), -(0, 2, 1), -(2, 1, 0)
    #   i.e.
    #
    #   - y^{0, 0} = \epsilon^{i1, j1, 0} \epsilon^{i2, j2, 0} xc1^{i1, i2} xc2^{j1, j2}
    #     Permutations: +(1, 2, 0), -(2, 1, 0);         +(1, 2, 0), -(2, 1, 0)
    #
    #   - y^{0, 1} = \epsilon^{i1, j1, 1} \epsilon^{i2, j2, 0} xc1^{i1, i2} xc2^{j1, j2}
    #     Permutations: +(2, 0, 1), -(0, 2, 1);         +(1, 2, 0), -(2, 1, 0)
    #
    #   - y^{0, 2} = \epsilon^{i1, j1, 2} \epsilon^{i2, j2, 0} xc1^{i1, i2} xc2^{j1, j2}
    #     Permutations: +(0, 1, 2), -(1, 0, 2)          +(1, 2, 0), -(2, 1, 0)
    #
    #   - y^{1, 0} = \epsilon^{i1, j1, 0} \epsilon^{i2, j2, 1} xc1^{i1, i2} xc2^{j1, j2}
    #     Permutations: +(1, 2, 0), -(2, 1, 0)          +(2, 0, 1), -(0, 2, 1)
    #
    #   - y^{1, 1} = \epsilon^{i1, j1, 1} \epsilon^{i2, j2, 1} xc1^{i1, i2} xc2^{j1, j2}
    #     Permutations: +(2, 0, 1), -(0, 2, 1)          +(2, 0, 1), -(0, 2, 1)
    #
    #   - y^{1, 2} = \epsilon^{i1, j1, 2} \epsilon^{i2, j2, 1} xc1^{i1, i2} xc2^{j1, j2}
    #     Permutations: +(0, 1, 2), -(1, 0, 2)          +(2, 0, 1), -(0, 2, 1)
    #
    #   - y^{2, 0} = \epsilon^{i1, j1, 0} \epsilon^{i2, j2, 2} xc1^{i1, i2} xc2^{j1, j2}
    #     Permutations: +(1, 2, 0), -(2, 1, 0)          +(0, 1, 2), -(1, 0, 2)
    #
    #   - y^{2, 1} = \epsilon^{i1, j1, 1} \epsilon^{i2, j2, 2} xc1^{i1, i2} xc2^{j1, j2}
    #     Permutations: +(2, 0, 1), -(0, 2, 1)          +(0, 1, 2), -(1, 0, 2)
    #
    #   - y^{2, 2} = \epsilon^{i1, j1, 2} \epsilon^{i2, j2, 2} xc1^{i1, i2} xc2^{j1, j2}
    #     Permutations: +(0, 1, 2), -(1, 0, 2)          +(0, 1, 2), -(1, 0, 2)
    #

    c1 = g.separate_color(msc1)
    c2 = g.separate_color(msc2)

    y = {}

    y[0, 0] = g.eval(c1[1, 1] * c2[2, 2] - c1[1, 2] * c2[2, 1] - c1[2, 1] * c2[1, 2] + c1[2, 2] * c2[1, 1])

    y[0, 1] = g.eval(c1[2, 1] * c2[0, 2] - c1[2, 2] * c2[0, 1] - c1[0, 1] * c2[2, 2] + c1[0, 2] * c2[2, 1])

    y[0, 2] = g.eval(c1[0, 1] * c2[1, 2] - c1[0, 2] * c2[1, 1] - c1[1, 1] * c2[0, 2] + c1[1, 2] * c2[0, 1])

    y[1, 0] = g.eval(c1[1, 2] * c2[2, 0] - c1[1, 0] * c2[2, 2] - c1[2, 2] * c2[1, 0] + c1[2, 0] * c2[1, 2])

    y[1, 1] = g.eval(c1[2, 2] * c2[0, 0] - c1[2, 0] * c2[0, 2] - c1[0, 2] * c2[2, 0] + c1[0, 0] * c2[2, 2])

    y[1, 2] = g.eval(c1[0, 2] * c2[1, 0] - c1[0, 0] * c2[1, 2] - c1[1, 2] * c2[0, 0] + c1[1, 0] * c2[0, 2])

    y[2, 0] = g.eval(c1[1, 0] * c2[2, 1] - c1[1, 1] * c2[2, 0] - c1[2, 0] * c2[1, 1] + c1[2, 1] * c2[1, 0])

    y[2, 1] = g.eval(c1[2, 0] * c2[0, 1] - c1[2, 1] * c2[0, 0] - c1[0, 0] * c2[2, 1] + c1[0, 1] * c2[2, 0])

    y[2, 2] = g.eval(c1[0, 0] * c2[1, 1] - c1[0, 1] * c2[1, 0] - c1[1, 0] * c2[0, 1] + c1[1, 1] * c2[0, 0])

    dst = g.mcolor(grid)
    g.merge_color(dst, y)

    return dst
