#
#    GPT - Grid Python Toolkit
#    Copyright (C) 2020  Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)
#                  2020  Stefan Meinel
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
#
#    Laplace operator:
#
#      laplace(U,D) \psi(x) = \sum_{\mu \in D} (U_\mu(x) \psi(x+\mu) + U_\mu^\dag(x-\mu) \psi(x-\mu) - 2 \psi(x))
#
#
#    Gaussian smearing operator:
#
#      gauss = ( 1 + sigma^2 / (4 steps) laplace )^steps
#
import gpt as g
from gpt.params import params_convention


def laplace(cov, dimensions):
    def mat(dst, src):
        assert dst != src
        dst[:] = 0.0
        for mu in dimensions:
            dst += g.eval(-2.0 * src + cov.forward[mu] * src + cov.backward[mu] * src)

    return g.matrix_operator(mat=mat)


@params_convention(
    boundary_phases=[1, 1, 1, -1], dimensions=[0, 1, 2], sigma=None, steps=None
)
def gauss(U, params):
    sigma = params["sigma"]
    steps = params["steps"]
    dimensions = params["dimensions"]
    lap = laplace(
        g.covariant.shift(U, boundary_phases=params["boundary_phases"]), dimensions
    )

    def mat(dst, src):
        assert dst != src
        g.copy(dst, src)
        for n in range(steps):
            dst += (sigma * sigma / (4.0 * steps)) * lap * dst

    return g.matrix_operator(mat=mat)
