#
#    GPT - Grid Python Toolkit
#    Copyright (C) 2021  Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)
#                  2021  Tom Blum
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
import numpy as np
from gpt.qcd.gauge.loops import accumulators, field_strength, default_rectangle_cache


def topological_charge(U, field=False, trace=True):
    assert len(U) == 4
    accumulator = accumulators[(field, trace)]
    res = accumulator(U[0])
    Bx = field_strength(U, 1, 2)
    By = field_strength(U, 2, 0)
    Bz = field_strength(U, 0, 1)
    Ex = field_strength(U, 3, 0)
    Ey = field_strength(U, 3, 1)
    Ez = field_strength(U, 3, 2)
    coeff = 8.0 / (32.0 * np.pi**2)
    coeff *= U[0].grid.gsites
    res += g(Bx * Ex + By * Ey + Bz * Ez)
    return res.scaled_project(coeff, True)


# O(a^4) improved def. of Q. See arXiv:hep-lat/9701012.
def topological_charge_5LI(U, field=False, trace=True, cache=default_rectangle_cache):
    assert len(U) == 4
    accumulator = accumulators[(field, trace)]
    c5 = 1 / 20.0
    c = [
        (19 - 55 * c5) / 9.0,
        (1 - 64 * c5) / 9.0,
        (-64 + 640 * c5) / 45.0,
        (1 / 5.0 - 2 * c5),
        c5,
    ]
    res = accumulator(U[0])
    # symmetric loops
    for loop, Lmu, Lnu in [(0, 1, 1), (1, 2, 2), (4, 3, 3)]:
        B = []
        E = []

        for mu, nu, T in [
            (1, 2, B),
            (2, 0, B),
            (0, 1, B),
            (3, 0, E),
            (3, 1, E),
            (3, 2, E),
        ]:
            A = g.qcd.gauge.rectangle(
                U,
                [
                    [
                        (mu, Lmu, nu, Lnu),
                        (nu, -Lnu, mu, Lmu),
                        (mu, -Lmu, nu, -Lnu),
                        (nu, Lnu, mu, -Lmu),
                    ]
                ],
                cache=cache,
                real=False,
                trace=False,
                field=True,
            )
            T.append(g(A - g.adj(A)))

        coeff = c[loop] / Lmu**2 / Lnu**2
        for i in range(0, 3):
            res += g(coeff * E[i] * B[i])

    # asymmetric loops
    for loop, Lmu, Lnu in [(2, 1, 2), (3, 1, 3)]:
        B = []
        E = []

        for mu, nu, T in [
            (1, 2, B),
            (2, 0, B),
            (0, 1, B),
            (3, 0, E),
            (3, 1, E),
            (3, 2, E),
        ]:
            A = g.qcd.gauge.rectangle(
                U,
                [
                    [
                        (mu, Lmu, nu, Lnu),
                        (nu, -Lnu, mu, Lmu),
                        (mu, -Lmu, nu, -Lnu),
                        (nu, Lnu, mu, -Lmu),
                        (mu, Lnu, nu, Lmu),
                        (nu, -Lmu, mu, Lnu),
                        (mu, -Lnu, nu, -Lmu),
                        (nu, Lmu, mu, -Lnu),
                    ]
                ],
                cache=cache,
                real=False,
                trace=False,
                field=True,
            )
            T.append(g(A - g.adj(A)))

        coeff = c[loop] / Lmu**2 / Lnu**2
        for i in range(0, 3):
            res += g(coeff * E[i] * B[i])

    # the first factor: 3 to remove rectangle norm by 3,
    # 2 because we need to avg over 4 * 2 clover leaves,
    # and rectangle only does 4.
    coeff = (3 / 2.0) ** 2 * 8.0 / (32.0 * np.pi**2)
    coeff *= U[0].grid.gsites
    return res.scaled_project(coeff, True)
