#
#    GPT - Grid Python Toolkit
#    Copyright (C) 2024  Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)
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

default_topological_charge_cache = {}


def topological_charge(
    U, field=False, trace=True, mask=None, cache=default_topological_charge_cache
):
    Nd = len(U)

    assert Nd == 4

    tag = f"{U[0].otype.__name__}_{U[0].grid}_{Nd}"

    if tag not in cache:
        code = []
        _target = 0
        _P = (0,) * Nd
        temporaries = [
            (0, 1, 2),  # Bx
            (1, 2, 0),  # By
            (2, 0, 1),  # Bz
            (3, 3, 0),  # Ex
            (4, 3, 1),  # Ey
            (5, 3, 2),  # Ez
        ]
        for tmp, mu, nu in temporaries:
            _temp1 = 1 + tmp
            code.append((_temp1, -1, 1.0, g.path().f(mu).f(nu).b(mu).b(nu)))
            code.append((_temp1, _temp1, -1.0, g.path().f(mu).b(nu).b(mu).f(nu)))
            code.append((_temp1, _temp1, 1.0, g.path().f(nu).b(mu).b(nu).f(mu)))
            code.append((_temp1, _temp1, -1.0, g.path().b(nu).b(mu).f(nu).f(mu)))
            code.append(
                (
                    _temp1,
                    _temp1,
                    -1.0,
                    [(_temp1, _P, 1)],
                )
            )

        coeff = 8.0 / (32.0 * np.pi**2) * (0.125**2.0)
        coeff *= U[0].grid.gsites

        for i in range(3):
            code.append(
                (
                    _target,
                    -1 if i == 0 else _target,
                    coeff,
                    [(1 + i, _P, 0), (4 + i, _P, 0)],
                )
            )

        cache[tag] = g.parallel_transport_matrix(U, code, 1)

    T = cache[tag](U)

    # return
    if trace:
        T = g(g.trace(T))

    if mask is not None:
        T *= mask

    if not field:
        T = g.sum(T).real / T.grid.gsites

    return T
