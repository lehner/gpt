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
from gpt.params import params_convention

verbose = g.default.is_verbose("staple_performance")

default_staple_cache = {}


def create_points_for_staple(Nd):
    points = []

    points.append((0,) * Nd)
    _P = 0

    evec = [np.array([1 if idx == j else 0 for idx in range(Nd)]) for j in range(Nd)]

    _Sp = [0] * Nd
    _Sm = [0] * Nd
    _Smp = [[0] * Nd for i in range(Nd)]

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

    return points, _P, _Sp, _Sm, _Smp


def staple_sum(U, rho, mu=None, cache=default_staple_cache):

    Nd = len(U)

    tag = f"{U[0].otype.__name__}_{U[0].grid}_{Nd}_{str(rho)}_{str(mu)}"

    if mu is None:
        target_mu = [i for i in range(Nd)]
    else:
        target_mu = [mu]

    Ntarget = len(target_mu)

    t = g.timer("staples")
    if tag not in cache:

        t("create")
        padding_U = g.padded_local_fields(U, [1] * Nd)
        padding_T = g.padded_local_fields([g.lattice(U[0]) for i in range(Ntarget)], [1] * Nd)

        padded_U = padding_U(U)

        Ntemp = 1  # need one temporary field for current staple precomputation

        _U = list(range(Ntarget + Ntemp, Ntarget + Ntemp + Nd))

        _temp = Ntarget

        code = []

        points, _P, _Sp, _Sm, _Smp = create_points_for_staple(Nd)

        nidx = [0] * Ntarget
        for idx in range(Ntarget):
            _mu = target_mu[idx]
            for _nu in range(Nd):
                if _nu != _mu:

                    code.append(
                        (
                            _temp,
                            -1,
                            1.0,
                            [(_U[_nu], _P, 0), (_U[_mu], _Sp[_nu], 0), (_U[_nu], _Sp[_mu], 1)],
                        )
                    )
                    code.append(
                        (
                            _temp,
                            _temp,
                            1.0,
                            [
                                (_U[_nu], _Sm[_nu], 1),
                                (_U[_mu], _Sm[_nu], 0),
                                (_U[_nu], _Smp[_nu][_mu], 0),
                            ],
                        )
                    )
                    # _temp now has mu-nu staple, now add it with right weights to targets
                    code.append(
                        (
                            idx,
                            -1 if nidx[idx] == 0 else idx,
                            complex(rho[idx, _nu]),
                            [(_temp, _P, 0)],
                        )
                    )
                    nidx[idx] += 1

        cache[tag] = (g.stencil.matrix(padded_U[0], points, code), padding_U, padding_T)
        t()

    c = cache[tag]

    t("halo")
    # halo exchange
    padded_U = c[1](U)
    padded_Temp = g.lattice(padded_U[0])
    padded_T = [g.lattice(padded_U[0]) for i in range(Ntarget)]

    t("stencil")
    # stencil staple sum calculation
    c[0](*padded_T, padded_Temp, *padded_U)

    t("bulk")
    # get bulk
    T = [g.lattice(U[0]) for i in range(Ntarget)]
    c[2].extract(T, padded_T)

    t()

    if verbose:
        g.message(t)
    # return
    return T
