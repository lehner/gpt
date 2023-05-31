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

default_staple_cache = {}


def staple_sum(U, rho, mu=None, cache=default_staple_cache):
    Nd = len(U)

    tag = f"{U[0].otype.__name__}_{U[0].grid}_{Nd}_{str(rho)}_{str(mu)}"

    if mu is None:
        target_mu = [i for i in range(Nd)]
    else:
        target_mu = [mu]

    Ntarget = len(target_mu)

    if tag not in cache:
        code = []
        nwr = [0] * Ntarget
        for idx in range(Ntarget):
            _mu = target_mu[idx]
            for _nu in range(Nd):
                if _nu != _mu:
                    code.append(
                        (
                            idx,
                            -1 if nwr[idx] == 0 else idx,
                            complex(rho[idx, _nu]),
                            g.path().f(_nu).f(_mu).b(_nu),
                        )
                    )
                    code.append((idx, idx, complex(rho[idx, _nu]), g.path().b(_nu).f(_mu).f(_nu)))
                    nwr[idx] = 1

        cache[tag] = g.parallel_transport_matrix(U, code, Ntarget)

    T = cache[tag](U)

    if Ntarget == 1:
        T = [T]

    return T
