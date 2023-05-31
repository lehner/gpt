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


def create_points_for_field_strength(Nd):
    points = []

    points.append((0,) * Nd)
    _P = 0

    evec = [np.array([1 if idx == j else 0 for idx in range(Nd)]) for j in range(Nd)]

    _Sp = [0] * Nd
    _Sm = [0] * Nd
    _Smp = [[0] * Nd for i in range(Nd)]
    _Smm = [[0] * Nd for i in range(Nd)]

    def _conv(x):
        return tuple([int(y) for y in x])

    for d in range(Nd):
        _Sp[d] = len(points)
        points.append(_conv(evec[d]))

        _Sm[d] = len(points)
        points.append(_conv(-evec[d]))

        for s in range(Nd):
            if s != d:
                _Smp[d][s] = len(points)
                points.append(_conv(evec[s] - evec[d]))

                _Smm[d][s] = len(points)
                points.append(_conv(-evec[s] - evec[d]))

    return points, _P, _Sp, _Sm, _Smp, _Smm


default_energy_density_cache = {}

def energy_density(U, field=False, trace=True, cache=default_energy_density_cache):

    Nd = len(U)

    tag = f"{U[0].otype.__name__}_{U[0].grid}_{Nd}"

    if tag not in cache:

        padding_U = g.padded_local_fields(U, [1] * Nd)
        padding_T = g.padded_local_fields(g.lattice(U[0]), [1] * Nd)

        padded_U = padding_U(U)

        Ntemp = 1

        Ntarget = 1

        _U = list(range(Ntarget + Ntemp, Ntarget + Ntemp + Nd))

        _temp1 = Ntarget

        _target = 0

        code = []

        points, _P, _Sp, _Sm, _Smp, _Smm = create_points_for_field_strength(Nd)

        nwr = 0
        for mu in range(Nd):
            for nu in range(mu):
                code.append(
                    (
                        _temp1,
                        -1,
                        1.0,
                        [(_U[mu], _P, 0), (_U[nu], _Sp[mu], 0), (_U[mu], _Sp[nu], 1), (_U[nu], _P, 1)],
                    )
                )
                code.append(
                    (
                        _temp1,
                        _temp1,
                        -1.0,
                        [(_U[mu], _P, 0), (_U[nu], _Smp[nu][mu], 1), (_U[mu], _Sm[nu], 1), (_U[nu], _Sm[nu], 0)],
                    )
                )

                code.append(
                    (
                        _temp1,
                        _temp1,
                        1.0,
                        [(_U[nu], _P, 0), (_U[mu], _Smp[mu][nu], 1), (_U[nu], _Sm[mu], 1), (_U[mu], _Sm[mu], 0)],
                    )
                )
                code.append(
                    (
                        _temp1,
                        _temp1,
                        -1.0,
                        [(_U[nu], _Sm[nu], 1), (_U[mu], _Smm[nu][mu], 1), (_U[nu], _Smm[nu][mu], 0), (_U[mu], _Sm[mu], 0)],
                    )
                )

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
                        -0.125**2.,
                        [(_temp1, _P, 0), (_temp1, _P, 0)],
                    )
                )
                nwr += 1
                

        cache[tag] = (g.stencil.matrix(padded_U[0], points, code), padding_U, padding_T)

    c = cache[tag]

    # halo exchange
    padded_U = c[1](U)
    padded_Temp = g.lattice(padded_U[0])
    padded_T = g.lattice(padded_U[0])

    # stencil staple sum calculation
    c[0](padded_T, padded_Temp, *padded_U)

    # get bulk
    T = g.lattice(U[0])
    c[2].extract(T, padded_T)

    # return
    if trace:
        T = g(g.trace(T))

    if not field:
        T = g.sum(T) / T.grid.gsites
        
    return T
