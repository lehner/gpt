#
#    GPT - Grid Python Toolkit
#    Copyright (C) 2022  Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)
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
import gpt as g


def diquark(Q1, Q2):
    eps = g.epsilon(Q1.otype.shape[2])
    R = g.lattice(Q1)

    # D_{a2,a1} = epsilon_{a1,b1,c1}*epsilon_{a2,b2,c2}*spin_transpose(Q1_{b1,b2})*Q2_{c1,c2}
    Q1 = g.separate_color(Q1)
    Q2 = g.separate_color(Q2)

    D = {x: g.lattice(Q1[x]) for x in Q1}
    for d in D:
        D[d][:] = 0

    for i1, sign1 in eps:
        for i2, sign2 in eps:
            D[i2[0], i1[0]] += (
                sign1 * sign2 * Q1[i1[1], i2[1]] * g.transpose(Q2[i1[2], i2[2]])
            )

    g.merge_color(R, D)
    return R
