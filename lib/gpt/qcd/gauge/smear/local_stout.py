#
#    GPT - Grid Python Toolkit
#    Copyright (C) 2020-24  Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)
#                           Adopted from Grid's GaugeConfigurationMasked with origins in Qlattice
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
import sys
from gpt.params import params_convention
from gpt.core.group import local_diffeomorphism, differentiable_functional


def compute_adj_ab(A, B, C, generators):
    # factors checked
    ng = len(generators)
    tmp = {}
    for b in range(ng):
        N_b = 2 * g.qcd.gauge.project.traceless_anti_hermitian(g.adj(A) * 1j * generators[b] * B)
        for c in range(ng):
            tmp[c, b] = g(-g.trace(1j * generators[c] * N_b))
    g.merge_color(C, tmp)


def compute_adj_abc(A, B, C, V, generators):
    # this is the bottle neck computationally right now ; TODO: make faster
    t = g.timer("compute_adj_abc")
    t("other")
    ng = len(generators)
    tmp = {}
    tmp2 = {}
    D = g.lattice(C)
    for a in range(ng):
        UtaU = g(g.adj(A) * 2j * generators[a] * B)
        for c in range(ng):
            fD = g.qcd.gauge.project.traceless_anti_hermitian(2j * generators[c] * UtaU)
            for d in range(ng):
                tmp[d, c] = g(-g.trace(1j * generators[d] * fD))
        t("merge")
        g.merge_color(D, tmp)
        t("other")
        tmp2[a,] = g(g.trace(C * D))
    t("merge")
    g.merge_color(V, tmp2)
    t()
    # print(t)


def csf(link, mu, field=None):
    if field is None:
        field = g.identity(link)
    return link * g.cshift(field, mu, 1)


def csb(link, mu, field=None):
    if field is None:
        field = g.identity(link)
    return g.cshift(g.adj(link) * field, mu, -1)


def adjoint_to_fundamental(fund, adj, generators):
    ng = len(generators)
    fund[:] = 0
    adj_c = g.separate_color(adj)
    for e in range(ng):
        fund += 1j * adj_c[e,] * generators[e]


class local_stout(local_diffeomorphism):
    @params_convention(dimension=None, checkerboard=None, rho=None)
    def __init__(self, params):
        self.params = params
        self.cache = {}

    def get_C(self, fields):
        grid = fields[0].grid
        nd = grid.nd
        U = fields[0:nd]
        rho = np.array(
            [[0.0 if (self.params["dimension"] == nu) else self.params["rho"] for nu in range(nd)]],
            dtype=np.float64,
        )

        if grid in self.cache:
            masks = self.cache[grid]
        else:
            grid_cb = grid.checkerboarded(g.redblack)
            one_cb = g.complex(grid_cb)
            one_cb[:] = 1

            masks = {}
            for p in [g.even, g.odd]:
                m = g.complex(grid)
                m[:] = 0
                one_cb.checkerboard(p)
                g.set_checkerboard(m, one_cb)
                masks[p] = m

            self.cache[grid] = masks

        mask, imask = masks[self.params["checkerboard"]], masks[self.params["checkerboard"].inv()]

        fm = g(mask + 1e-15 * imask)
        st = g.qcd.gauge.staple_sum(U, mu=self.params["dimension"], rho=rho)[0]
        return g(st * fm), U, fm

    def __call__(self, fields):
        C_mu, U, fm = self.get_C(fields)
        mu = self.params["dimension"]
        U_prime = g.copy(U)
        U_prime[mu] @= g(
            g.matrix.exp(g.qcd.gauge.project.traceless_anti_hermitian(C_mu * g.adj(U[mu]))) * U[mu]
        )
        return U_prime

    def jacobian(self, fields, fields_prime, src):
        nd = fields[0].grid.nd
        U_prime = fields_prime[0:nd]

        C_mu, U, fm = self.get_C(fields)

        assert len(src) == nd

        dst = [g.lattice(s) for s in src]

        # (75) of https://arxiv.org/pdf/hep-lat/0311018.pdf
        mu = self.params["dimension"]

        #
        # Sigma == g.adj(U) * gradient * 1j
        #
        Sigma_prime_mu = g(g.adj(U_prime[mu]) * src[mu] * 1j)
        U_Sigma_prime_mu = g(U[mu] * Sigma_prime_mu)

        iQ_mu = g.qcd.gauge.project.traceless_anti_hermitian(C_mu * g.adj(U[mu]))
        exp_iQ_mu, Lambda_mu = g.matrix.exp.function_and_gradient(iQ_mu, U_Sigma_prime_mu)

        Lambda_mu *= fm

        dst[mu] @= Sigma_prime_mu * exp_iQ_mu + g.adj(C_mu) * 1j * Lambda_mu

        for nu in range(nd):
            if nu != mu:
                dst[nu] @= g(g.adj(U_prime[nu]) * src[nu] * 1j)

        rho = self.params["rho"]

        for nu in range(nd):
            if mu == nu:
                continue

            U_nu_x_plus_mu = g.cshift(U[nu], mu, 1)
            U_mu_x_plus_nu = g.cshift(U[mu], nu, 1)
            Lambda_mu_x_plus_nu = g.cshift(Lambda_mu, nu, 1)

            dst[nu] -= 1j * rho * U_mu_x_plus_nu * g.adj(U_nu_x_plus_mu) * g.adj(U[mu]) * Lambda_mu

            dst[nu] += (
                1j
                * rho
                * Lambda_mu_x_plus_nu
                * U_mu_x_plus_nu
                * g.adj(U_nu_x_plus_mu)
                * g.adj(U[mu])
            )

            dst[mu] -= (
                1j
                * rho
                * U_nu_x_plus_mu
                * g.adj(U_mu_x_plus_nu)
                * Lambda_mu_x_plus_nu
                * g.adj(U[nu])
            )

            dst[mu] += g.cshift(
                -1j * rho * g.adj(U_nu_x_plus_mu) * g.adj(U[mu]) * Lambda_mu * U[nu],
                nu,
                -1,
            )

            dst[nu] += g.cshift(
                1j * rho * g.adj(U_mu_x_plus_nu) * g.adj(U[nu]) * Lambda_mu * U[mu]
                - 1j * rho * g.adj(U_mu_x_plus_nu) * Lambda_mu_x_plus_nu * g.adj(U[nu]) * U[mu],
                mu,
                -1,
            )

        for mu in range(nd):
            dst[mu] @= U[mu] * dst[mu] * (-1j)
            dst[mu] @= g.qcd.gauge.project.traceless_hermitian(dst[mu])

        return dst

    def jacobian_components(self, fields):
        C_mu, U, fm = self.get_C(fields)
        mu = self.params["dimension"]

        U_mu = U[mu]

        grid = U_mu.grid
        dt = grid.precision.complex_dtype
        otype = fields[0].otype
        cartesian_otype = otype.cartesian()
        adjoint_otype = g.ot_matrix_su_n_adjoint_algebra(otype.Nc)
        generators = cartesian_otype.generators(dt)
        ng = len(generators)

        N_cb = g.lattice(grid, adjoint_otype)
        Z_ac = g.lattice(grid, adjoint_otype)

        adjoint_generators = adjoint_otype.generators(dt)

        M = g(U_mu * g.adj(C_mu))

        adj_id = g.identity(g.lattice(grid, adjoint_otype))
        fund_id = g.identity(g.lattice(grid, otype))

        compute_adj_ab(fund_id, M, N_cb, generators)

        Z = g(g.qcd.gauge.project.traceless_anti_hermitian(g.adj(M)))

        Z_ac[:] = 0
        for b in range(ng):
            coeff = g(2 * g.trace(1j * generators[b] * Z))
            Z_ac += 1j * adjoint_generators[b] * coeff

        # compute J
        X = g.copy(adj_id)
        J_ac = g.copy(adj_id)
        kpfac = 1.0
        denom = g.norm2(X)
        nmax = 25
        for k in range(1, nmax):
            X @= X * Z_ac
            kpfac = kpfac / (k + 1)
            Y = g(X * kpfac)
            eps = (g.norm2(Y) / denom) ** 0.5
            J_ac += Y
            if eps < grid.precision.eps:
                break
        assert k != nmax - 1

        # combined M
        M_ab = g(adj_id - J_ac * N_cb)

        # return component
        return J_ac, N_cb, Z_ac, M, fm, M_ab

    def log_det_jacobian(self, fields):
        J_ac, N_cb, Z_ac, M, fm, M_ab = self.jacobian_components(fields)
        det_M = g.matrix.det(M_ab)
        log_det_M = g(g.component.real(g.component.log(det_M)))
        log_det = g(fm * log_det_M)
        return log_det

    def action_log_det_jacobian(self):
        return local_stout_action_log_det_jacobian(self)


class local_stout_action_log_det_jacobian(differentiable_functional):
    def __init__(self, stout):
        self.stout = stout

    def __call__(self, U):
        log_det = g.sum(self.stout.log_det_jacobian(U))
        return -log_det.real

    def gradient(self, U, dU):
        assert dU == U

        J_ac, NxxAd, Z_ac, M, fm, M_ab = self.stout.jacobian_components(U)

        grid = J_ac.grid
        dtype = grid.precision.complex_dtype
        otype = U[0].otype.cartesian()
        adjoint_otype = J_ac.otype
        adjoint_vector_otype = g.ot_vector_color(adjoint_otype.Ndim)
        adjoint_generators = adjoint_otype.generators(dtype)
        generators = otype.generators(dtype)
        ng = len(adjoint_generators)

        rho = self.stout.params["rho"]

        one = g.complex(grid)
        one[:] = 1

        t = g.timer("action_log_det_jacobian")
        t("dJdX")

        # dJdX
        dJdX = [g(1j * adjoint_generators[b] * one) for b in range(ng)]
        aunit = g.identity(J_ac)

        X = g.copy(Z_ac)
        t2 = g.copy(X)
        for j in reversed(range(2, 13)):
            t3 = g(t2 * (1 / (j + 1)) + aunit)
            t2 @= X * t3
            for b in range(ng):
                dJdX[b] = 1j * adjoint_generators[b] * t3 + X * dJdX[b] * (1 / (j + 1))

        for b in range(ng):
            dJdX[b] = g(-dJdX[b])

        t("Invert M_ab")
        inv_M_ab = g.matrix.inv(M_ab)

        t("N M^-1")
        nMpInv = g(NxxAd * inv_M_ab)
        MpInvJx = g((-1.0) * inv_M_ab * J_ac)

        PlaqL = g.identity(U[0])
        PlaqR = g(M * fm)
        FdetV = g.lattice(grid, adjoint_vector_otype)
        compute_adj_abc(PlaqL, PlaqR, MpInvJx, FdetV, generators)

        Fdet2_mu = g.copy(FdetV)
        Fdet1_mu = g(0 * FdetV)

        tmp = {}
        for e in range(ng):
            tmp[e,] = g(g.trace(dJdX[e] * nMpInv))
        dJdXe_nMpInv = g.lattice(grid, adjoint_vector_otype)
        g.merge_color(dJdXe_nMpInv, tmp)
        dJdXe_nMpInv @= dJdXe_nMpInv * fm

        mu = self.stout.params["dimension"]
        U_mu_masked = g(U[mu] * fm)

        # fundamental forces
        Fdet1 = [g.lattice(grid, otype) for nu in range(len(U))]
        Fdet2 = [g.lattice(grid, otype) for nu in range(len(U))]

        t("non-local")
        Nxy = g.lattice(NxxAd)
        for nu in range(len(U)):
            if nu == mu:
                continue

            # + nu cw
            PlaqL = g.identity(U[0])
            PlaqR = g((-rho) * csf(U[nu], nu, csf(U[mu], mu, csb(U[nu], nu, csb(U_mu_masked, mu)))))

            dJdXe_nMpInv_y = dJdXe_nMpInv
            compute_adj_ab(PlaqL, PlaqR, Nxy, generators)
            Fdet1_nu = g(g.transpose(Nxy) * dJdXe_nMpInv_y)

            PlaqR = g((-1.0) * PlaqR)
            t("compute_adj_abc")
            compute_adj_abc(PlaqL, PlaqR, MpInvJx, FdetV, generators)
            t("non-local")
            Fdet2_nu = g.copy(FdetV)

            # + nw acw
            PlaqR = g(rho * csf(U[nu], nu, csb(U[mu], mu, csb(U[nu], nu))))
            PlaqL = csb(U_mu_masked, mu)

            dJdXe_nMpInv_y = g.cshift(dJdXe_nMpInv, mu, -1)
            compute_adj_ab(PlaqL, PlaqR, Nxy, generators)
            Fdet1_nu += g.transpose(Nxy) * dJdXe_nMpInv_y

            MpInvJx_nu = g.cshift(MpInvJx, mu, -1)
            t("compute_adj_abc")
            compute_adj_abc(PlaqL, PlaqR, MpInvJx_nu, FdetV, generators)
            t("non-local")
            Fdet2_nu += FdetV

            # - nu cw
            PlaqL = g(rho * csf(U[mu], mu, csf(U[nu], nu, csb(U_mu_masked, mu))))
            PlaqR = csf(U[nu], nu)

            dJdXe_nMpInv_y = g.cshift(dJdXe_nMpInv, nu, 1)
            compute_adj_ab(PlaqL, PlaqR, Nxy, generators)
            Fdet1_nu += g.transpose(Nxy) * dJdXe_nMpInv_y

            MpInvJx_nu = g.cshift(MpInvJx, nu, 1)
            t("compute_adj_abc")
            compute_adj_abc(PlaqL, PlaqR, MpInvJx_nu, FdetV, generators)
            t("non-local")
            Fdet2_nu += FdetV

            # -nu acw
            PlaqL = g((-rho) * csf(U[nu], nu, csb(U_mu_masked, mu)))
            PlaqR = csb(U[mu], mu, csf(U[nu], nu))

            dJdXe_nMpInv_y = g.cshift(dJdXe_nMpInv, mu, -1)
            dJdXe_nMpInv_y = g.cshift(dJdXe_nMpInv_y, nu, 1)

            compute_adj_ab(PlaqL, PlaqR, Nxy, generators)
            Fdet1_nu += g.transpose(Nxy) * dJdXe_nMpInv_y

            MpInvJx_nu = g.cshift(MpInvJx, mu, -1)
            MpInvJx_nu = g.cshift(MpInvJx_nu, nu, 1)
            t("compute_adj_abc")
            compute_adj_abc(PlaqL, PlaqR, MpInvJx_nu, FdetV, generators)
            t("non-local")
            Fdet2_nu += FdetV

            # force contributions to fundamental representation
            adjoint_to_fundamental(Fdet1[nu], Fdet1_nu, generators)
            adjoint_to_fundamental(Fdet2[nu], Fdet2_nu, generators)

            # mu cw
            PlaqL = g((-rho) * csf(U[mu], mu, csb(U[nu], nu, csb(U_mu_masked, mu))))
            PlaqR = csb(U[nu], nu)

            dJdXe_nMpInv_y = g.cshift(dJdXe_nMpInv, nu, -1)
            compute_adj_ab(PlaqL, PlaqR, Nxy, generators)
            Fdet1_mu += g.transpose(Nxy) * dJdXe_nMpInv_y

            MpInvJx_nu = g.cshift(MpInvJx, nu, -1)
            t("compute_adj_abc")
            compute_adj_abc(PlaqL, PlaqR, MpInvJx_nu, FdetV, generators)
            t("non-local")
            Fdet2_mu += FdetV

            # mu acw
            PlaqL = g((-rho) * csf(U[mu], mu, csf(U[nu], nu, csb(U_mu_masked, mu))))
            PlaqR = csf(U[nu], nu)

            dJdXe_nMpInv_y = g.cshift(dJdXe_nMpInv, nu, 1)

            compute_adj_ab(PlaqL, PlaqR, Nxy, generators)
            Fdet1_mu += g.transpose(Nxy) * dJdXe_nMpInv_y

            MpInvJx_nu = g.cshift(MpInvJx, nu, 1)

            t("compute_adj_abc")
            compute_adj_abc(PlaqL, PlaqR, MpInvJx_nu, FdetV, generators)
            t("non-local")
            Fdet2_mu += FdetV

        t("aggregate")
        Fdet1_mu += g.transpose(NxxAd) * dJdXe_nMpInv

        adjoint_to_fundamental(Fdet1[mu], Fdet1_mu, generators)
        adjoint_to_fundamental(Fdet2[mu], Fdet2_mu, generators)

        force = [g((0.5 * 1j) * (x + y)) for x, y in zip(Fdet1, Fdet2)]

        t()
        print(t)
        return force
