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


default_plaquette_cache = {}


def plaquette(U, cache=default_plaquette_cache):

    vol = float(U[0].grid.fsites)
    Nd = len(U)
    ndim = U[0].otype.shape[0]

    tag = f"{U[0].otype.__name__}_{U[0].grid}_{Nd}"

    if tag not in cache:

        padding_U = g.padded_local_fields(U, [1] * Nd)
        padding = g.padded_local_fields(U[0], [1] * Nd)

        padded_U = padding_U(U)

        _P = 0
        _U = list(range(1, 1 + Nd))
        _Sp = list(range(1, 1 + Nd))

        code = []

        for mu in range(Nd):
            for nu in range(mu):
                code.append(
                    (
                        0,
                        -1 if len(code) == 0 else 0,
                        1.0,
                        [
                            (_U[mu], _P, 0),
                            (_U[nu], _Sp[mu], 0),
                            (_U[mu], _Sp[nu], 1),
                            (_U[nu], _P, 1),
                        ],
                    )
                )

        cache[tag] = (
            g.stencil.matrix(
                padded_U[0],
                [  # (0,0,0,0), (1,0,0,0), (0,1,0,0), ...
                    tuple([1 if idx == mu else 0 for mu in range(Nd)]) for idx in range(-1, Nd)
                ],
                code,
            ),
            padding_U,
            padding,
        )

    c = cache[tag]

    # halo exchange
    padded_U = c[1](U)
    padded_P = g.lattice(padded_U[0])

    # stencil plaquette calculation
    c[0](padded_P, *padded_U)

    # get bulk
    P = g.lattice(U[0])
    c[2].extract(P, padded_P)

    # extract plaquette
    return 2 * g.sum(g.trace(P)).real / vol / Nd / (Nd - 1) / ndim
