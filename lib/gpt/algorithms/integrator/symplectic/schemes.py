#
#    GPT - Grid Python Toolkit
#    Copyright (C) 2020-25  Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)
#                  2020  Mattia Bruno
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
from gpt.algorithms.integrator.symplectic import symplectic_base


def complete_coefficients(r):
    x0 = sum(r[0::2])
    x1 = sum(r[1::2])
    r.append(0.5 - x0)
    r.append(0.5 - x1)


def generic(N, ia, r, q, tag):
    s = symplectic_base(f"{tag}({N}, {ia[0].__name__}, {ia[1].__name__})")

    for j in range(N):
        for i in range(len(r) // 2):
            s.add(q[2 * i + 0], r[2 * i + 0] / N, 0)
            s.add(q[2 * i + 1], r[2 * i + 1] / N, 0)

        for i in reversed(range(len(r) // 2)):
            s.add(q[2 * i + 1], r[2 * i + 1] / N, 0)
            s.add(q[2 * i + 0], r[2 * i + 0] / N, 0)

    s.simplify()

    s.insert(*ia)

    s.simplify()

    s.add_directions()

    return s


# force-gradient integrators
def OMF2_force_gradient(N, i0, i1, ifg, l1=1.0 / 6.0, l2=0.5):
    # https://arxiv.org/pdf/0910.2950
    r = [l1, l2]
    q = [0, 1, 2, 3]  # 3 is never used in this scheme
    complete_coefficients(r)
    ifg = ifg(2.0 / 72.0, 2.0 * r[-2])
    return generic(N, [i0, i1, ifg], r, q, f"OMF2_force_gradient({l1},{l2})")


# force integrators
def OMF4(N, i0, i1):
    # Omelyan, Mryglod, Folk, 4th order integrator
    #   ''Symplectic analytically integrable decomposition algorithms ...''
    #   https://doi.org/10.1016/S0010-4655(02)00754-3
    #      values of r's can be found @ page 292, sec 3.5.1, Variant 8
    r = [
        0.08398315262876693,
        0.2539785108410595,
        0.6822365335719091,
        -0.03230286765269967,
    ]
    complete_coefficients(r)
    q = [0, 1, 0, 1, 0, 1]
    return generic(N, [i0, i1], r, q, "OMF4")


def OMF2(N, i0, i1, l1=0.18, l2=0.5):
    r = [l1, l2]
    complete_coefficients(r)
    q = [0, 1, 0, 1]
    return generic(N, [i0, i1], r, q, f"OMF2({l1},{l2})")


def leap_frog(N, i0, i1):
    return generic(N, [i0, i1], [0.5, 0.5], [0, 1], "leap_frog")
