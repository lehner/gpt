#
#    GPT - Grid Python Toolkit
#    Copyright (C) 2020  Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)
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
from gpt.params import params_convention


def staple(U, mu, nu):
    assert mu != nu
    U_nu_x_plus_mu = g.cshift(U[nu], mu, 1)
    U_mu_x_plus_nu = g.cshift(U[mu], nu, 1)
    U_nu_x_minus_nu = g.cshift(U[nu], nu, -1)
    return g(
        U[nu] * U_mu_x_plus_nu * g.adj(U_nu_x_plus_mu)
        + g.adj(U_nu_x_minus_nu) * g.cshift(U[mu] * U_nu_x_plus_mu, nu, -1)
    )


@params_convention(rho=None)
def staple_sum(U, params):
    nd = len(U)
    rho = params["rho"]
    assert rho is not None
    assert rho.shape == (nd, nd)
    U_prime = []
    for mu in range(nd):
        U_mu_prime = g.lattice(U[mu])
        U_mu_prime[:] = 0
        for nu in range(nd):
            if mu != nu:
                if abs(rho[mu, nu]) != 0.0:
                    U_mu_prime += rho[mu, nu] * staple(U, mu, nu)
        U_prime.append(U_mu_prime)
    return U_prime
