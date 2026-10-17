#
#    GPT - Grid Python Toolkit
#    Copyright (C) 2026  Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)
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
# Building blocks for the generators of directional_frozen_shift: functions of
# the frozen links F (a list of nd links, U_mu masked) that work on plain
# fields and on reverse-AD nodes and are gauge covariant.  mu (the updated
# direction) and the update mask P1 are bound at construction.
#
import gpt as g
from gpt.core.parallel_transport.matrix import new_target_list
from gpt.qcd.gauge.smear.directional_parallel_transport import _link_offsets, _shifted


def _directions(nd, mu):
    # the transverse directions (all for mu = None)
    return [nu for nu in range(nd) if nu != mu]


def _key(x):
    return str(type(x))


def _transport(cache, links, code, n):
    # the n outputs of one fused parallel_transport_matrix (plain fields or
    # nodes), cached per link type
    key = _key(links[0])
    if key not in cache:
        cache[key] = g.parallel_transport_matrix(links, code, n)
    ptm = cache[key]
    if n > 1 and isinstance(links[0], g.ad.reverse.node_base):
        # (keep the list node: parallel_transport_matrix returns its elements)
        T = new_target_list(links[0], n)
        ptm.stencil(T, *links)
        return [T[k] for k in range(n)]
    return g.util.to_list(ptm(links))


class transverse_staples:
    # C_nu(x) = sum_{lam != nu, mu} [V_lam(x) V_nu(x + lam) V_lam(x + nu)^dag +
    # V_lam(x - lam)^dag V_nu(x - lam) V_lam(x - lam + nu)] for nu != mu, the
    # staples of the links V_nu in the planes transverse to mu (they never read
    # V_mu); one fused stencil.  Returns the list of the C_nu (in the order of
    # the directions nu != mu).  mu = None: all directions (the 4d staples)
    def __init__(self, mu):
        self.mu = mu
        self.cache = {}

    def __call__(self, F):
        nd = len(F)
        dirs = _directions(nd, self.mu)
        code = []
        for k, nu in enumerate(dirs):
            first = True
            for lam in dirs:
                if lam == nu:
                    continue
                code.append((k, -1 if first else k, 1.0, g.path().f(lam).f(nu).b(lam)))
                code.append((k, k, 1.0, g.path().b(lam).f(nu).f(lam)))
                first = False
        return _transport(self.cache, F, code, len(dirs))


class _transverse_flow:
    def __init__(self, mu, checkpoint=None):
        # checkpoint = k: on nodes, every k steps form one g.ad.reverse.checkpoint
        # (the graph keeps the links between the segments only; the backward
        # re-runs a segment), None: no checkpoints
        self.mu = mu
        self.staples = transverse_staples(mu)
        assert checkpoint is None or checkpoint >= 1
        self.checkpoint = checkpoint

    def _steps(self, F, n):
        raise NotImplementedError()

    def __call__(self, F):
        k = self.checkpoint
        if k is None or not any(isinstance(x, g.ad.reverse.node_base) for x in F):
            return self._steps(F, self.n)
        done = 0
        while done < self.n:
            m = min(k, self.n - done)
            F = g.ad.reverse.checkpoint(lambda X, m=m: self._steps(X, m), F)
            done += m
        return F

    def Z(self, F):
        # TA(C_nu V_nu^dag) for nu != mu (the Euler direction of the Wilson
        # flow in the transverse planes, see transverse_stout)
        dirs = _directions(len(F), self.mu)
        C = self.staples(F)
        return [g.qcd.gauge.project.traceless_anti_hermitian(g(c * g.adj(F[nu]))) for c, nu in zip(C, dirs)]

    def _step(self, F, Z):
        # V_nu -> exp(Z_nu) V_nu for nu != mu
        dirs = _directions(len(F), self.mu)
        F = list(F)
        for z, nu in zip(Z, dirs):
            F[nu] = g(g.matrix.exp(z) * F[nu])
        return F


class transverse_stout(_transverse_flow):
    # n steps of stout smearing of the links nu != mu in the planes transverse
    # to mu, V_nu -> exp(rho TA(C_nu V_nu^dag)) V_nu (as g.qcd.gauge.smear.stout
    # with orthogonal_dimension = mu, but without reading V_mu); returns the
    # list of nd links with F[mu] unchanged; checkpoint = k: see
    # _transverse_flow
    def __init__(self, mu, rho, n=1, checkpoint=None):
        super().__init__(mu, checkpoint)
        self.rho = rho
        self.n = n

    def _steps(self, F, n):
        for _ in range(n):
            F = self._step(F, [g(self.rho * z) for z in self.Z(F)])
        return F


class transverse_wilson_flow(_transverse_flow):
    # n steps (step size epsilon) of the Wilson flow of the links nu != mu in
    # the planes transverse to mu, by Luescher's third-order Runge-Kutta scheme
    # (arXiv:1006.4518, appendix C) with Z(V) = TA(C V^dag) (so an Euler step
    # is transverse_stout with rho = epsilon); returns the list of nd links
    # with F[mu] unchanged; checkpoint = k: see _transverse_flow
    def __init__(self, mu, epsilon, n=1, checkpoint=None):
        super().__init__(mu, checkpoint)
        self.epsilon = epsilon
        self.n = n

    def _steps(self, F, n):
        e = self.epsilon
        for _ in range(n):
            Z0 = self.Z(F)
            W1 = self._step(F, [g((e / 4) * z0) for z0 in Z0])
            Z1 = self.Z(W1)
            W2 = self._step(W1, [g(e * ((8 / 9) * z1 - (17 / 36) * z0)) for z0, z1 in zip(Z0, Z1)])
            Z2 = self.Z(W2)
            F = self._step(
                W2,
                [g(e * ((3 / 4) * z2 - (8 / 9) * z1 + (17 / 36) * z0)) for z0, z1, z2 in zip(Z0, Z1, Z2)],
            )
        return F


def _path_end(path, nd):
    pos = [0] * nd
    for nu, d in path.path:
        pos[nu] += d
    return tuple(pos)


class frozen_transporters:
    # the parallel transports of the frozen links along paths from x that
    # end at x + mu (transporters x -> x + mu) or at x (closed loops), in one
    # fused stencil; returns the list of the transports.  The constructor
    # rejects (ValueError) a path that traverses U_mu(x + s) with x + s in P1
    # for some x in P1 (an updated link, P1(x) P1(x + s) != 0), and a path
    # with another end point
    def __init__(self, mu, P1, paths):
        self.mu = mu
        self.paths = list(paths)
        nd = P1.grid.nd
        overlap = {}
        e_mu = tuple(1 if i == mu else 0 for i in range(nd))
        for k, p in enumerate(self.paths):
            end = _path_end(p, nd)
            if end != e_mu and end != (0,) * nd:
                raise ValueError(f"frozen_transporters: path {k} ends at {end}, not at x or x + mu")
            offsets, _ = _link_offsets(p, nd, mu)
            for s in offsets:
                if s not in overlap:
                    overlap[s] = g.norm2(P1 * _shifted(P1, s)) > 0
                if overlap[s]:
                    raise ValueError(
                        f"frozen_transporters: path {k} traverses the link U_{mu}(x + {s}), "
                        "which the step updates for some x in P1"
                    )
        self.code = [(k, -1, 1.0, p) for k, p in enumerate(self.paths)]
        self.cache = {}

    def __call__(self, F):
        return _transport(self.cache, F, self.code, len(self.paths))


def staple_paths(mu, nu):
    # the staples x -> x + mu through x + nu and through x - nu
    return [g.path().f(nu).f(mu).b(nu), g.path().b(nu).f(mu).f(nu)]


def staple_sum_paths(mu, nd):
    return [p for nu in _directions(nd, mu) for p in staple_paths(mu, nu)]


class transverse_green:
    # w = (m2 - Delta_perp)^-1 s for a site field s in the adjoint
    # (s -> V s V^dag; e.g. an algebra field), with Delta_perp the covariant
    # Laplacian over the directions nu != mu built from the links F[nu]:
    #
    #   (m2 - Delta_perp) w = (m2 + 2 (nd - 1)) w
    #       - sum_nu [V_nu(x) w(x + nu) V_nu(x)^dag + V_nu(x - nu)^dag w(x - nu) V_nu(x - nu)]
    #
    # (hermitian and positive for m2 > 0: solver e.g. cg), one matrix stencil.
    # The solve is g.ad.reverse.linear_solve: differentiable w.r.t. F and s
    # (first order)
    def __init__(self, mu, m2, solver):
        assert m2 > 0
        self.mu = mu
        self.m2 = m2
        self.solver = solver
        self.stencil = None

    def _stencil(self, w):
        if self.stencil is None:
            nd = w.grid.nd
            dirs = _directions(nd, self.mu)
            n = len(dirs)
            origin = (0,) * nd
            ev = [tuple(1 if i == nu else 0 for i in range(nd)) for nu in dirs]
            points = [origin] + ev + [tuple(-x for x in e) for e in ev]
            _w = 1 + n
            code = [(0, -1, self.m2 + 2 * n, [(_w, 0, 0)])]
            for k in range(n):
                _V = 1 + k
                code.append((0, 0, -1.0, [(_V, 0, 0), (_w, 1 + k, 0), (_V, 0, 1)]))
                code.append((0, 0, -1.0, [(_V, 1 + n + k, 1), (_w, 1 + n + k, 0), (_V, 1 + n + k, 0)]))
            self.stencil = g.stencil.matrix(w, points, code)
        return self.stencil

    def operator(self, *args):
        # (m2 - Delta_perp) w for args = (V_nu for nu != mu) + (w,); plain or
        # nodes
        *V, w = args
        st = self._stencil(w)
        if any(isinstance(x, g.ad.reverse.node_base) for x in args):
            V = [g.ad.reverse.util.constant(x) for x in V]
            w = g.ad.reverse.util.constant(w)
            out = w.new()
        else:
            out = g.lattice(w)
        st(out, *V, w)
        return out

    def __call__(self, F, s):
        V = [F[nu] for nu in _directions(len(F), self.mu)]
        return g.ad.reverse.linear_solve(self.operator, s, self.solver, parameters=V)


class staple_pair_generator:
    # rho S_+ S_-^dag with S_+- the staples x -> x + mu through x +- nu: the
    # simplest covariant generator (a test generator; rho a number, or None:
    # the first parameter)
    def __init__(self, mu, P1, nu, rho=None):
        self.transporters = frozen_transporters(mu, P1, staple_paths(mu, nu))
        self.rho = rho

    def __call__(self, F, xparams):
        up, dn = self.transporters(F)
        rho = xparams[0] if self.rho is None else self.rho
        return g(rho * up * g.adj(dn))


class smoothed_staple_generator:
    # rho C_raw M^dag with C_raw the staple sum x -> x + mu of the frozen
    # links and M the staple sum of the links after smooth (e.g.
    # transverse_stout(mu, ...), which changes only the links nu != mu); a
    # test generator (rho a number, or None: the first parameter)
    def __init__(self, mu, P1, smooth, rho=None):
        self.transporters = frozen_transporters(mu, P1, staple_sum_paths(mu, P1.grid.nd))
        self.smooth = smooth
        self.rho = rho

    def _sum(self, F):
        T = self.transporters(F)
        s = T[0]
        for t in T[1:]:
            s = s + t
        return s

    def __call__(self, F, xparams):
        C = self._sum(F)
        M = self._sum(self.smooth(F))
        rho = xparams[0] if self.rho is None else self.rho
        return g(rho * C * g.adj(M))
