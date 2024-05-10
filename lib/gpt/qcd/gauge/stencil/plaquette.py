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
        code = []
        for mu in range(Nd):
            for nu in range(mu):
                code.append((0, -1 if len(code) == 0 else 0, 1.0, g.path().f(mu).f(nu).b(mu).b(nu)))

        cache[tag] = g.parallel_transport_matrix(U, code, 1)

    P = cache[tag](U)

    return 2 * g.sum(g.trace(P)).real / vol / Nd / (Nd - 1) / ndim
