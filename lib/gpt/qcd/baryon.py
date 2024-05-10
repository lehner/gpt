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


default_cache = {}


def diquark(Q1, Q2, cache=default_cache):
    R = g.lattice(Q1)
    # D_{a2,a1} = epsilon_{a1,b1,c1}*epsilon_{a2,b2,c2}*Q1_{b1,b2}*spin_transpose(Q2_{c1,c2})
    cache_key = f"{Q1.otype.__name__}_{Q1.checkerboard().__name__}_{Q1.grid.describe()}"
    if cache_key not in cache:
        Nc = Q1.otype.shape[2]
        Ns = Q1.otype.shape[0]
        eps = g.epsilon(Nc)
        code = []
        acc = {}
        ti = g.stencil.tensor_instructions
        for i in range(Ns):
            for j in range(Ns):
                for l in range(Ns):
                    for i1, sign1 in eps:
                        for i2, sign2 in eps:
                            dst = (i * Ns + j) * Nc * Nc + i2[0] * Nc + i1[0]
                            aa = (Ns * i + l) * Nc * Nc + i1[1] * Nc + i2[1]
                            bb = (Ns * j + l) * Nc * Nc + i1[2] * Nc + i2[2]
                            if dst not in acc:
                                acc[dst] = True
                                mode = ti.mov if sign1 * sign2 > 0 else ti.mov_neg
                            else:
                                mode = ti.inc if sign1 * sign2 > 0 else ti.dec
                            code.append((0, dst, mode, 1.0, [(1, 0, aa), (2, 0, bb)]))

        segments = [(len(code) // (Ns * Ns), Ns * Ns)]
        cache[cache_key] = g.stencil.tensor(Q1, [(0, 0, 0, 0)], code, segments)

    cache[cache_key](R, Q1, Q2)
    return R
