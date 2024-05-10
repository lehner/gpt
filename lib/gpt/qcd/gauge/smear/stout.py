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
import numpy as np
import gpt as g
from gpt.params import params_convention
from gpt.core.group import diffeomorphism


def get_rho(U, params):
    rho = params["rho"]
    if isinstance(rho, np.ndarray):
        return rho

    nd = len(U)
    orthogonal_dimension = params["orthogonal_dimension"]
    if orthogonal_dimension is None:
        orthogonal_dimension = -1

    return np.array(
        [
            [
                0.0
                if (mu == orthogonal_dimension or nu == orthogonal_dimension or mu == nu)
                else rho
                for nu in range(nd)
            ]
            for mu in range(nd)
        ],
        dtype=np.float64,
    )


class stout(diffeomorphism):
    @params_convention(rho=None, orthogonal_dimension=None)
    def __init__(self, params):
        self.params = params

    # apply the smearing
    def __call__(self, fields):
        nd = fields[0].grid.nd
        U = fields[0:nd]
        C = g.qcd.gauge.staple_sum(U, rho=get_rho(U, self.params))
        U_prime = []
        for mu in range(nd):
            U_mu_prime = g(
                g.matrix.exp(g.qcd.gauge.project.traceless_anti_hermitian(C[mu] * g.adj(U[mu])))
                * U[mu]
            )
            U_prime.append(U_mu_prime)

        return U_prime + fields[nd:]

    # apply the jacobian
    def jacobian(self, fields, fields_prime, src):
        nd = fields[0].grid.nd
        U = fields[0:nd]
        U_prime = fields_prime[0:nd]

        rho = get_rho(U, self.params)
        C = g.qcd.gauge.staple_sum(U, rho=rho)

        assert len(src) == nd

        dst = [g.lattice(s) for s in src]
        exp_iQ = [None] * nd
        Lambda = [None] * nd
        Sigma_prime = [None] * nd

        # (75) of https://arxiv.org/pdf/hep-lat/0311018.pdf
        for mu in range(nd):
            #
            # Sigma == g.adj(U) * gradient * 1j
            #
            Sigma_prime[mu] = g(g.adj(U_prime[mu]) * src[mu] * 1j)
            U_Sigma_prime_mu = g(U[mu] * Sigma_prime[mu])

            iQ_mu = g.qcd.gauge.project.traceless_anti_hermitian(C[mu] * g.adj(U[mu]))
            exp_iQ[mu], Lambda[mu] = g.matrix.exp.function_and_gradient(iQ_mu, U_Sigma_prime_mu)

            dst[mu] @= Sigma_prime[mu] * exp_iQ[mu] + g.adj(C[mu]) * 1j * Lambda[mu]

        for mu in range(nd):
            for nu in range(nd):
                if mu == nu:
                    continue

                rho_mu_nu = rho[mu, nu]
                rho_nu_mu = rho[nu, mu]

                if abs(rho_nu_mu) != 0.0 or abs(rho_mu_nu) != 0.0:
                    U_nu_x_plus_mu = g.cshift(U[nu], mu, 1)
                    U_mu_x_plus_nu = g.cshift(U[mu], nu, 1)
                    Lambda_nu_x_plus_mu = g.cshift(Lambda[nu], mu, 1)
                    Lambda_mu_x_plus_nu = g.cshift(Lambda[mu], nu, 1)

                    dst[mu] -= (
                        1j
                        * rho_nu_mu
                        * U_nu_x_plus_mu
                        * g.adj(U_mu_x_plus_nu)
                        * g.adj(U[nu])
                        * Lambda[nu]
                    )

                    dst[mu] += (
                        1j
                        * rho_nu_mu
                        * Lambda_nu_x_plus_mu
                        * U_nu_x_plus_mu
                        * g.adj(U_mu_x_plus_nu)
                        * g.adj(U[nu])
                    )

                    dst[mu] -= (
                        1j
                        * rho_mu_nu
                        * U_nu_x_plus_mu
                        * g.adj(U_mu_x_plus_nu)
                        * Lambda_mu_x_plus_nu
                        * g.adj(U[nu])
                    )

                    dst[mu] += g.cshift(
                        1j * rho_nu_mu * g.adj(U_nu_x_plus_mu) * g.adj(U[mu]) * Lambda[nu] * U[nu]
                        - 1j * rho_mu_nu * g.adj(U_nu_x_plus_mu) * g.adj(U[mu]) * Lambda[mu] * U[nu]
                        - 1j
                        * rho_nu_mu
                        * g.adj(U_nu_x_plus_mu)
                        * Lambda_nu_x_plus_mu
                        * g.adj(U[mu])
                        * U[nu],
                        nu,
                        -1,
                    )

        for mu in range(nd):
            dst[mu] @= U[mu] * dst[mu] * (-1j)
            dst[mu] @= g.qcd.gauge.project.traceless_hermitian(dst[mu])

        return dst
