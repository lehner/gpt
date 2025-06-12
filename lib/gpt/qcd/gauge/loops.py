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
import numpy as np

default_rectangle_cache = {}


class accumulator_base:
    def scaled_project(self, scale, real):
        if g.util.is_num(self.value):
            return scale * (self.value.real if real else self.value)
        else:
            if real:
                return g((g.adj(self.value) + self.value) * (scale / 2.0))
            else:
                return g(self.value * scale)


class accumulator_average(accumulator_base):
    def __init__(self, like):
        self.value = g.tensor(np.zeros(like.otype.shape, dtype=np.complex128), like.otype)

    def __iadd__(self, v):
        v = g(v)
        self.value += g.sum(v) / v.grid.gsites
        return self


class accumulator_average_trace(accumulator_base):
    def __init__(self, like):
        self.value = 0.0

    def __iadd__(self, v):
        v = g(g.trace(v))
        self.value += g.sum(v) / v.grid.gsites
        return self


class accumulator_field(accumulator_base):
    def __init__(self, like):
        self.value = g.lattice(like)
        self.value[:] = 0.0

    def __iadd__(self, v):
        self.value += v
        return self


class accumulator_field_trace(accumulator_base):
    def __init__(self, like):
        self.value = g.complex(like.grid)
        self.value[:] = 0.0

    def __iadd__(self, v):
        self.value += g.trace(v)
        return self


accumulators = {
    (False, False): accumulator_average,
    (False, True): accumulator_average_trace,
    (True, False): accumulator_field,
    (True, True): accumulator_field_trace,
}


def rectangle(
    U,
    first,
    second=None,
    third=None,
    cache=default_rectangle_cache,
    field=False,
    trace=True,
    real=True,
):
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
    accumulator = accumulators[(field, trace)]
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
                    [(mu, L_mu, nu, L_nu) for mu in range(min_mu, len(U)) for nu in range(mu)]
                )
            else:
                configurations.append(f)

    cache_key = f"{len(U)}_{U[0].otype.__name__}_{U[0].grid}_" + str(configurations)
    if cache_key not in cache:
        paths = []
        elements = []
        for configuration in configurations:
            c_paths = [
                g.path().f(mu, L_mu).f(nu, L_nu).b(mu, L_mu).b(nu, L_nu)
                for mu, L_mu, nu, L_nu in configuration
            ]
            elements.append(len(c_paths))
            paths = paths + c_paths
        cache[cache_key] = (g.parallel_transport(U, paths), elements)

    transport = cache[cache_key][0]
    ranges = cache[cache_key][1]

    loops = transport(U)
    ndim = U[0].otype.shape[0]
    value = accumulator(U[0])
    idx = 0
    ridx = 0
    results = []
    for p in loops:
        value += p
        idx += 1
        if idx == ranges[ridx]:
            results.append(value.scaled_project(1.0 / idx / ndim, real))
            idx = 0
            ridx = ridx + 1
            value = accumulator(U[0])
    if len(results) == 1:
        return results[0]
    return results


def differentiable_staple(U, mu, nu):
    staple_up = g.cshift(U[nu], mu, 1) * g.adj(g.cshift(U[mu], nu, 1)) * g.adj(U[nu])
    staple_down = g.cshift(g.adj(g.cshift(U[nu], mu, 1)) * g.adj(U[mu]) * U[nu], nu, -1)
    return staple_up, staple_down


def field_strength(U, mu, nu):
    assert mu != nu
    staple_up, staple_down = differentiable_staple(U, mu, nu)
    v = g(staple_up - staple_down)
    F = g.eval(U[mu] * v + g.cshift(v * U[mu], mu, -1))
    F = 0.125 * (F - g.adj(F))
    return F


def differentiable_topology(aU):
    Bx = field_strength(aU, 1, 2)
    By = field_strength(aU, 2, 0)
    Bz = field_strength(aU, 0, 1)

    Ex = field_strength(aU, 3, 0)
    Ey = field_strength(aU, 3, 1)
    Ez = field_strength(aU, 3, 2)

    coeff = 8.0 / (32.0 * np.pi**2)

    Q = g.sum(g.trace(Bx * Ex) + g.trace(By * Ey) + g.trace(Bz * Ez)) * coeff

    return Q


def differentiable_energy_density(aU):
    Nd = len(aU)
    grid = aU[0].grid
    res = None
    for mu in range(Nd):
        for nu in range(mu):
            Fmunu = field_strength(aU, mu, nu)
            if res is None:
                res = Fmunu * Fmunu
            else:
                res += Fmunu * Fmunu
    return (-1.0 / grid.gsites) * g.sum(g.trace(res))


def differentiable_P_and_R(aU):
    Nd = len(aU)
    grid = aU[0].grid
    ndim = aU[0].otype.shape[0]
    res_P = None
    res_R = None
    for mu in range(Nd):
        for nu in range(Nd):
            if mu == nu:
                continue

            staple_up, staple_down = differentiable_staple(aU, mu, nu)

            P = g.sum(g.trace(aU[mu] * staple_up))
            R = g.sum(g.trace(g.adj(staple_down) * staple_up))

            res_P = P if res_P is None else P + res_P
            res_R = R if res_R is None else R + res_R

    res_P = (res_P + g.adj(res_P)) * (0.5 / (Nd - 1) / Nd / ndim / grid.gsites)
    res_R = (res_R + g.adj(res_R)) * (0.5 / (Nd - 1) / Nd / ndim / grid.gsites)
    return res_P, res_R
