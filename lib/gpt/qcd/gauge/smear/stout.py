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
                (
                    0.0
                    if (mu == orthogonal_dimension or nu == orthogonal_dimension or mu == nu)
                    else rho
                )
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
        self.verbose = g.default.is_verbose("stout_performance")
        self.stencil = None

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

        tt = g.timer("jacobian")

        tt("rho")
        rho = get_rho(U, self.params)

        tt("staple sum")
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
            tt("expr")
            Sigma_prime[mu] = g(g.adj(U_prime[mu]) * src[mu] * 1j)
            U_Sigma_prime_mu = g(U[mu] * Sigma_prime[mu])

            tt("traceless anti hermitian")
            iQ_mu = g.qcd.gauge.project.traceless_anti_hermitian(C[mu] * g.adj(U[mu]))

            tt("exponential")
            exp_iQ[mu], Lambda[mu] = g.matrix.exp.function_and_gradient(iQ_mu, U_Sigma_prime_mu)

            tt("expr")
            dst[mu] @= Sigma_prime[mu] * exp_iQ[mu] + g.adj(C[mu]) * 1j * Lambda[mu]

        # create all shifted U and lambda fields at once
        U_shifted = [g.lattice(U[0]) for mu in range(nd * nd)]
        Lambda_shifted = [g.lattice(Lambda[0]) for mu in range(nd * nd)]
        if self.stencil is None:
            directions = [tuple([1 if mu == nu else 0 for mu in range(nd)]) for nu in range(nd)]
            self.stencil = g.stencil.matrix(
                U[0],
                directions,
                [
                    (nd * mu + nu, -1, 1.0, [(mu + nd * nd * 2, nu, 0)])
                    for mu in range(nd)
                    for nu in range(nd)
                    if mu != nu
                ]
                + [
                    (nd * mu + nu + nd * nd, -1, 1.0, [(mu + nd * nd * 2 + nd, nu, 0)])
                    for mu in range(nd)
                    for nu in range(nd)
                    if mu != nu
                ],
            )
        self.stencil(*U_shifted, *Lambda_shifted, *U, *Lambda)

        for mu in range(nd):
            for nu in range(nd):
                if mu == nu:
                    continue

                rho_mu_nu = rho[mu, nu]
                rho_nu_mu = rho[nu, mu]

                if abs(rho_nu_mu) != 0.0 or abs(rho_mu_nu) != 0.0:
                    tt("cshift")
                    U_nu_x_plus_mu = U_shifted[nd * nu + mu]
                    U_mu_x_plus_nu = U_shifted[nd * mu + nu]
                    Lambda_nu_x_plus_mu = Lambda_shifted[nd * nu + mu]
                    Lambda_mu_x_plus_nu = Lambda_shifted[nd * mu + nu]

                    tt("expr")

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

                    tt("cshift")
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

                    tt()

        if self.verbose:
            g.message(tt)

        for mu in range(nd):
            dst[mu] @= U[mu] * dst[mu] * (-1j)
            dst[mu] @= g.qcd.gauge.project.traceless_hermitian(dst[mu])

        return dst


class differentiable_stout:
    def __init__(self, rho):
        self.rho = rho

    def __call__(self, aU):
        nd = len(aU)
        C = [None] * nd
        for mu in range(nd):
            for nu in range(nd):
                if nu == mu:
                    continue
                su, sd = g.qcd.gauge.differentiable_staple(aU, mu, nu)
                c = g.adj(su + sd)
                if C[mu] is None:
                    C[mu] = c
                else:
                    C[mu] = C[mu] + c
            C[mu] = self.rho * C[mu]

        U_prime = []
        for mu in range(nd):
            U_mu_prime = (
                g.matrix.exp(g.qcd.gauge.project.traceless_anti_hermitian(C[mu] * g.adj(aU[mu])))
                * aU[mu]
            )
            U_prime.append(U_mu_prime)

        return U_prime
