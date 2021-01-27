#
#    GPT - Grid Python Toolkit
#    Copyright (C) 2020  Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)
#                  2020  Tilo Wettig
#                  2020  Simon Buerger
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

default_rectangle_cache = {}


def rectangle(U, first, second=None, third=None, cache=default_rectangle_cache):
    #
    # Calling conventions:
    #
    # rectangle(U, 2, 1)
    # rectangle(U, 2, 1, 3)  # fixes the temporal extent to be L_mu and averages over spatial sizes L_nu
    # rectangle(U, [(1,1), (2,1)])
    #
    # or specify explicit mu,L_mu,nu,L_nu configurations
    # rectangle(U, [ [ (0,1,3,2), (1,2,3,2) ] ])
    #
    if second is not None:
        L_mu = first
        L_nu = second
        min_mu = third if third is not None else 0
        configurations = [
            [(mu, L_mu, nu, L_nu) for mu in range(min_mu, len(U)) for nu in range(mu)]
        ]
    else:
        configurations = []
        for f in first:
            if type(f) is tuple:
                L_mu = f[0]
                L_nu = f[1]
                min_mu = f[2] if len(f) == 3 else 0
                configurations.append(
                    [
                        (mu, L_mu, nu, L_nu)
                        for mu in range(min_mu, len(U))
                        for nu in range(mu)
                    ]
                )
            else:
                configurations.append(f)

    cache_key = str(configurations)
    if cache_key not in cache:
        paths = []
        elements = []
        for configuration in configurations:
            c_paths = [
                g.qcd.gauge.path().f(mu, L_mu).f(nu, L_nu).b(mu, L_mu).b(nu, L_nu)
                for mu, L_mu, nu, L_nu in configuration
            ]
            elements.append(len(c_paths))
            paths = paths + c_paths
        cache[cache_key] = (g.qcd.gauge.transport(U, paths), elements)

    transport = cache[cache_key][0]
    ranges = cache[cache_key][1]

    loops = transport(U)

    vol = float(U[0].grid.fsites)
    ndim = U[0].otype.shape[0]

    value = 0.0
    idx = 0
    ridx = 0
    results = []
    for p in loops:
        value += g.sum(g.trace(p))
        idx += 1
        if idx == ranges[ridx]:
            results.append(value.real / vol / idx / ndim)
            idx = 0
            ridx = ridx + 1
            value = 0.0
    if len(results) == 1:
        return results[0]
    return results


def plaquette(U):
    # U[mu](x)*U[nu](x+mu)*adj(U[mu](x+nu))*adj(U[nu](x))
    tr = 0.0
    vol = float(U[0].grid.fsites)
    Nd = len(U)
    ndim = U[0].otype.shape[0]
    for mu in range(Nd):
        for nu in range(mu):
            tr += g.sum(
                g.trace(
                    U[mu]
                    * g.cshift(U[nu], mu, 1)
                    * g.adj(g.cshift(U[mu], nu, 1))
                    * g.adj(U[nu])
                )
            )
    return 2.0 * tr.real / vol / Nd / (Nd - 1) / ndim


def field_strength(U, mu, nu):
    assert mu != nu
    # v = staple_up - staple_down
    v = g.eval(
        g.cshift(U[nu], mu, 1) * g.adj(g.cshift(U[mu], nu, 1)) * g.adj(U[nu])
        - g.cshift(g.adj(g.cshift(U[nu], mu, 1)) * g.adj(U[mu]) * U[nu], nu, -1)
    )

    F = g.eval(U[mu] * v + g.cshift(v * U[mu], mu, -1))
    F @= 0.125 * (F - g.adj(F))
    return F
