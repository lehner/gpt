#
#    GPT - Grid Python Toolkit
#    Copyright (C) 2023  Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)
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

default_energy_density_cache = {}


def energy_density(U, field=False, trace=True, cache=default_energy_density_cache):
    Nd = len(U)

    tag = f"{U[0].otype.__name__}_{U[0].grid}_{Nd}"

    if tag not in cache:
        code = []
        _temp1 = 1
        _target = 0
        nwr = 0
        _P = (0,) * Nd
        for mu in range(Nd):
            for nu in range(mu):
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
                code.append(
                    (
                        _target,
                        -1 if nwr == 0 else _target,
                        -(0.125**2.0),
                        [(_temp1, _P, 0), (_temp1, _P, 0)],
                    )
                )
                nwr += 1

        cache[tag] = g.parallel_transport_matrix(U, code, 1)

    T = cache[tag](U)

    # return
    if trace:
        T = g(g.trace(T))

    if not field:
        T = g.sum(T) / T.grid.gsites

    return T
