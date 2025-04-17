#
#    GPT - Grid Python Toolkit
#    Copyright (C) 2025  Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)
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


def create_stencil_operator_n_rhs(points, ip, n_rhs, ocb, packed_right_hand_sides, args):
    assert packed_right_hand_sides is None
    assert args == []
    
    def _mat(dst, src):
        for i in range(len(src)):
            dst[i][:] = 0
            for p in points:
                src_i = src[i]
                for mu in range(len(p)):
                    if p[mu] != 0:
                        src_i = g.cshift(src_i, mu, p[mu])
                assert dst[i].checkerboard() == src_i.checkerboard()
                assert points[p].checkerboard() == src_i.checkerboard()
                dst[i] += points[p] * src_i

    return g.matrix_operator(_mat, accept_list=(True, True))
