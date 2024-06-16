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

local_stout_parallel_projector = g.default.get_int("--local-stout-parallel-projector", 1)

plaquette_stencil_cache = {}


def create_adjoint_projector(D, B, generators, nfactors):
    ng = len(generators)
    code = []
    idst = 0
    if nfactors == 1:
        iB = 1
    else:
        iA = 1
        iB = 2
    igen = iB + 1
    ndim = B.otype.shape[0]
    ti = g.stencil.tensor_instructions
    for c in range(ng):
        if local_stout_parallel_projector:
            itmp1 = -(igen + ng + 2 * c)
            itmp2 = -(igen + ng + 2 * c + 1)
        else:
            itmp1 = -(igen + ng)
            itmp2 = -(igen + ng + 1)
        # itmp1 = 2j * generators[c] * B
        imm = itmp1 if nfactors == 1 else itmp2
        ti.matrix_multiply(code, ndim, 2j, imm, igen + c, iB)
        if nfactors == 2:
            ti.matrix_multiply(code, ndim, 1.0, itmp1, iA, itmp2)
        # itmp2 = 0.5 * itmp1 - 0.5 * adj(itmp1)
        ti.matrix_anti_hermitian(code, ndim, itmp2, itmp1)
        # itmp1[0,0] = g.trace(itmp2) / 3.0
        ti.matrix_trace(code, ndim, 0, 1.0 / 3.0, itmp1, itmp2)
        # itmp2[i,i] -= itmp1[0,0]
        ti.matrix_diagonal_subtract(code, ndim, itmp2, itmp1)
        # now Dprime[d, c] = g(-g.trace(1j * generators[d] * itmp2))
        for d in range(ng):
            dst = d * ng + c
            ti.matrix_trace_ab(code, ndim, dst, -1j, idst, igen + d, itmp2)

    if local_stout_parallel_projector:
        segments = [(len(code) // ng, ng)]
    else:
        segments = [(len(code) // 1, 1)]
    ein = g.stencil.tensor(D, [(0, 0, 0, 0)], code, segments)

    nx = 2 * ng if local_stout_parallel_projector else 2
    fgenerators = [g.lattice(B) for d in range(ng + nx)]
    for d in range(ng):
        fgenerators[d][:] = generators[d]

    return ein, fgenerators


def adjoint_from_right_fast(D, UtaU, generators, cache):
    if "stencil" not in cache:
        cache["stencil"] = create_adjoint_projector(D, UtaU, generators, 1)

    ein, fgenerators = cache["stencil"]

    ein(D, UtaU, *fgenerators)


def compute_adj_ab(A, B, C, generators, cache):
    if "stencil_ab" not in cache:
        cache["stencil_ab"] = create_adjoint_projector(C, A, generators, 2)

    ein, fgenerators = cache["stencil_ab"]

    ein(C, g(g.adj(A)), g(B), *fgenerators)


def compute_adj_abc(_A, _B, _C, _V, generators, cache, parity):

    t = g.timer("compute_adj_abc")
    t("checkerboarding")
    A = g.pick_checkerboard(parity, _A)
    B = g.pick_checkerboard(parity, _B)
    C = g.pick_checkerboard(parity, _C)
    V = g.pick_checkerboard(parity, _V)

    t("other")
    ng = len(generators)
    tmp2 = {}
    D = g.lattice(C)
    for a in range(ng):
        UtaU = g(g.adj(A) * 2j * generators[a] * B)

        # move the loop below to a stencil.tensor
        t("adj_from_right")
        adjoint_from_right_fast(D, UtaU, generators, cache)

        t("other")
        tmp2[a,] = g(g.trace(C * D))
    t("merge")
    g.merge_color(V, tmp2)
    t("checkerboarding")

    _V[:] = 0
    g.set_checkerboard(_V, V)

    t()
    # g.message(t)


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
        self.verbose = g.default.is_verbose("stout_performance")

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

    def inv(self, fields, max_iter=100):
        C_mu, U, fm = self.get_C(fields)
        mu = self.params["dimension"]
        U_prime = g.copy(U)
        for it in range(max_iter):
            U_prime_mu_last = g.copy(U_prime[mu])
            U_prime[mu] @= g(
                g.matrix.exp(
                    -g.qcd.gauge.project.traceless_anti_hermitian(C_mu * g.adj(U_prime[mu]))
                )
                * U[mu]
            )
            eps2 = g.norm2(U_prime_mu_last - U_prime[mu]) / U_prime_mu_last.grid.gsites
            if eps2 < U_prime_mu_last.grid.precision.eps**2:
                break
        if it == max_iter - 1:
            # indicate failure
            g.message(
                f"Warning: local_stout could not be inverted; last eps^2 = {eps2} after {max_iter} iterations"
            )
            return None
        return U_prime

    def jacobian(self, fields, fields_prime, src):
        nd = fields[0].grid.nd
        U_prime = fields_prime[0:nd]

        t = g.timer("local_stout_jacobian")

        t("local")
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

            t("non-local")
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

        t("local")
        for mu in range(nd):
            dst[mu] @= U[mu] * dst[mu] * (-1j)
            dst[mu] @= g.qcd.gauge.project.traceless_hermitian(dst[mu])

        t()
        if self.verbose:
            g.message(t)

        return dst

    def jacobian_components(self, fields, cache_ab):
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

        compute_adj_ab(fund_id, M, N_cb, generators, cache_ab)

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
        cache_ab = {}
        J_ac, N_cb, Z_ac, M, fm, M_ab = self.jacobian_components(fields, cache_ab)
        det_M = g.matrix.det(M_ab)
        log_det_M = g(g.component.real(g.component.log(det_M)))
        log_det = g(fm * log_det_M)
        return log_det

    def action_log_det_jacobian(self):
        return local_stout_action_log_det_jacobian(self)


class local_stout_action_log_det_jacobian(differentiable_functional):
    def __init__(self, stout):
        self.stout = stout
        self.verbose = stout.verbose

    def plaquette_stencil(self, U, rho, mu, nu):
        key = f"{U[0].grid.describe()}_{U[0].otype.__name__}_{mu}_{nu}_{rho}"
        if key not in self.stout.cache:
            code = []
            code.append((0, -1, -rho, g.path().f(nu).f(mu).b(nu).b(mu)))
            code.append((1, -1, -rho, g.path().f(nu).b(mu)))
            code.append((2, -1, -rho, g.path().f(mu).b(nu).b(mu)))
            code.append((3, -1, -rho, g.path().f(mu).f(nu).b(mu)))
            code.append((4, -1, rho, g.path().f(nu).b(mu).b(nu)))
            code.append((5, -1, rho, g.path().f(mu).f(nu).b(mu)))
            code.append((6, -1, 1.0, g.path().b(mu)))
            code.append((7, -1, 1.0, g.path().b(nu)))
            code.append((8, -1, 1.0, g.path().f(nu)))
            code.append((9, -1, 1.0, g.path().b(mu).f(nu)))

            self.stout.cache[key] = g.parallel_transport_matrix(U, code, 10)

        return self.stout.cache[key](U)

    def __call__(self, U):
        log_det = g.sum(self.stout.log_det_jacobian(U))
        return -log_det.real

    def gradient(self, U, dU):
        assert dU == U

        t = g.timer("action_log_det_jacobian")

        cb = self.stout.params["checkerboard"]

        t("jac_comp")
        cache_ab = {}
        J_ac, NxxAd, Z_ac, M, fm, M_ab = self.stout.jacobian_components(U, cache_ab)

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

        t("dJdX")

        # dJdX -> stencil version
        dJdX = [g(1j * adjoint_generators[b] * one) for b in range(ng)]
        aunit = g.identity(J_ac)

        X = g.copy(Z_ac)
        t2 = g.copy(X)
        for j in reversed(range(2, 13)):
            t3 = g(t2 * (1 / (j + 1)) + aunit)
            t2 @= X * t3
            for b in range(ng):
                dJdX[b] @= 1j * adjoint_generators[b] * t3 + X * dJdX[b] * (1 / (j + 1))

        for b in range(ng):
            dJdX[b] = g(-dJdX[b])

        t("invert M_ab")
        inv_M_ab = g.matrix.inv(M_ab)

        t("N M^-1")
        nMpInv = g(NxxAd * inv_M_ab)
        MpInvJx = g((-1.0) * inv_M_ab * J_ac)

        PlaqL = g.identity(U[0])
        PlaqR = g(M * fm)
        FdetV = g.lattice(grid, adjoint_vector_otype)
        cache = {}

        compute_adj_abc(PlaqL, PlaqR, MpInvJx, FdetV, generators, cache, cb)

        Fdet2_mu = g.copy(FdetV)
        Fdet1_mu = g(0 * FdetV)

        tmp = {}
        for e in range(ng):
            tmp[e,] = g(g.trace(dJdX[e] * nMpInv))
        dJdXe_nMpInv = g.lattice(grid, adjoint_vector_otype)
        g.merge_color(dJdXe_nMpInv, tmp)
        dJdXe_nMpInv @= dJdXe_nMpInv * fm

        mu = self.stout.params["dimension"]

        # fundamental forces
        Fdet1 = [g.lattice(grid, otype) for nu in range(len(U))]
        Fdet2 = [g.lattice(grid, otype) for nu in range(len(U))]

        t("non-local cshift")
        Nxy = g.lattice(NxxAd)

        dJdXe_nMpInv_bmu = g.cshift(dJdXe_nMpInv, mu, -1)
        MpInvJx_bmu = g.cshift(MpInvJx, mu, -1)

        for nu in range(len(U)):
            if nu == mu:
                continue

            t("non-local stencil")

            (
                minus_fnu_fmu_bnu_bmu,
                minus_fnu_bmu,
                minus_fmu_bnu_bmu,
                minus_fmu_fnu_bmu,
                plus_fnu_bmu_bnu,
                plus_fmu_fnu_bmu,
                one_bmu,
                one_bnu,
                one_fnu,
                one_bmu_fnu,
            ) = self.plaquette_stencil(U, rho, mu, nu)

            for cb_field in [minus_fnu_fmu_bnu_bmu, minus_fnu_bmu]:
                cb_field @= cb_field * fm

            for icb_field in [minus_fmu_bnu_bmu, minus_fmu_fnu_bmu, plus_fmu_fnu_bmu, one_bmu]:
                icb_field @= icb_field * (one - fm)

            dJdXe_nMpInv_bnu = g.cshift(dJdXe_nMpInv, nu, -1)
            dJdXe_nMpInv_fnu = g.cshift(dJdXe_nMpInv, nu, 1)
            dJdXe_nMpInv_fnu_bmu = g.cshift(dJdXe_nMpInv_bmu, nu, 1)
            MpInvJx_fnu = g.cshift(MpInvJx, nu, 1)
            MpInvJx_fnu_bmu = g.cshift(MpInvJx_bmu, nu, 1)
            MpInvJx_bnu = g.cshift(MpInvJx, nu, -1)

            # + nu cw
            PlaqL = g.identity(U[0])
            PlaqR = minus_fnu_fmu_bnu_bmu

            t("compute_adj_ab")
            compute_adj_ab(PlaqL, PlaqR, Nxy, generators, cache_ab)
            Fdet1_nu = g(g.transpose(Nxy) * dJdXe_nMpInv)

            PlaqR = g((-1.0) * PlaqR)
            t("compute_adj_abc")

            compute_adj_abc(PlaqL, PlaqR, MpInvJx, FdetV, generators, cache, cb)
            Fdet2_nu = g.copy(FdetV)

            # + nw acw
            PlaqR = plus_fnu_bmu_bnu
            PlaqL = one_bmu

            t("compute_adj_ab")
            compute_adj_ab(PlaqL, PlaqR, Nxy, generators, cache_ab)
            Fdet1_nu += g.transpose(Nxy) * dJdXe_nMpInv_bmu

            t("compute_adj_abc")
            compute_adj_abc(PlaqL, PlaqR, MpInvJx_bmu, FdetV, generators, cache, cb.inv())
            Fdet2_nu += FdetV

            # - nu cw
            PlaqL = plus_fmu_fnu_bmu
            PlaqR = one_fnu

            t("compute_adj_ab")
            compute_adj_ab(PlaqL, PlaqR, Nxy, generators, cache_ab)
            Fdet1_nu += g.transpose(Nxy) * dJdXe_nMpInv_fnu

            t("compute_adj_abc")
            compute_adj_abc(PlaqL, PlaqR, MpInvJx_fnu, FdetV, generators, cache, cb.inv())
            Fdet2_nu += FdetV

            # -nu acw
            PlaqL = minus_fnu_bmu
            PlaqR = one_bmu_fnu

            t("compute_adj_ab")
            compute_adj_ab(PlaqL, PlaqR, Nxy, generators, cache_ab)
            Fdet1_nu += g.transpose(Nxy) * dJdXe_nMpInv_fnu_bmu

            t("compute_adj_abc")
            compute_adj_abc(PlaqL, PlaqR, MpInvJx_fnu_bmu, FdetV, generators, cache, cb)
            Fdet2_nu += FdetV

            # force contributions to fundamental representation
            t("adj_to_fund")
            adjoint_to_fundamental(Fdet1[nu], Fdet1_nu, generators)
            adjoint_to_fundamental(Fdet2[nu], Fdet2_nu, generators)

            # mu cw
            PlaqL = minus_fmu_bnu_bmu
            PlaqR = one_bnu

            t("compute_adj_ab")
            compute_adj_ab(PlaqL, PlaqR, Nxy, generators, cache_ab)
            Fdet1_mu += g.transpose(Nxy) * dJdXe_nMpInv_bnu

            t("compute_adj_abc")
            compute_adj_abc(PlaqL, PlaqR, MpInvJx_bnu, FdetV, generators, cache, cb.inv())
            Fdet2_mu += FdetV

            # mu acw
            PlaqL = minus_fmu_fnu_bmu
            PlaqR = one_fnu

            t("compute_adj_ab")
            compute_adj_ab(PlaqL, PlaqR, Nxy, generators, cache_ab)
            Fdet1_mu += g.transpose(Nxy) * dJdXe_nMpInv_fnu

            t("compute_adj_abc")
            compute_adj_abc(PlaqL, PlaqR, MpInvJx_fnu, FdetV, generators, cache, cb.inv())
            Fdet2_mu += FdetV

        t("aggregate")
        Fdet1_mu += g.transpose(NxxAd) * dJdXe_nMpInv

        t("adj_to_fund")
        adjoint_to_fundamental(Fdet1[mu], Fdet1_mu, generators)
        adjoint_to_fundamental(Fdet2[mu], Fdet2_mu, generators)
        t("aggregate")

        force = [g((0.5 * 1j) * (x + y)) for x, y in zip(Fdet1, Fdet2)]

        t()
        if self.verbose:
            g.message(t)
        return force
