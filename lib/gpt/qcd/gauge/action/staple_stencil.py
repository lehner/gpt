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
import gpt as g

rad = g.ad.reverse


def staple_stencil_code(Nd, wP, wR, rectangles):
    # plaquettes and 2x1 rectangles from the up and down staples, which are
    # local temporaries of the stencil (computed once per site and shared by
    # the plaquette and the rectangles):
    #   T_up[mu,nu](x) = U_nu(x+mu) U_mu(x+nu)^dag U_nu(x)^dag
    #   T_dn[mu,nu](x) = U_nu(x+mu-nu)^dag U_mu(x-nu)^dag U_nu(x-nu)
    #   O(x) = sum_{mu != nu} [ wP/2 U_mu(x) T_up + wR T_dn^dag T_up ]
    # (every plaquette appears twice among the ordered pairs, every 2x1
    # rectangle once).  Fields: 0 = O, then the temporaries, then the links.
    pairs = [(mu, nu) for mu in range(Nd) for nu in range(Nd) if mu != nu]
    z = (0,) * Nd

    def e(*steps):
        p = [0] * Nd
        for d, s in steps:
            p[d] += s
        return tuple(p)

    TU = {pr: 1 + k for k, pr in enumerate(pairs)}
    TD = {pr: 1 + len(pairs) + k for k, pr in enumerate(pairs)} if rectangles else {}
    nT = len(TU) + len(TD)
    L = lambda mu: 1 + nT + mu
    code = []
    for mu, nu in pairs:
        code.append((TU[mu, nu], -1, 1.0, [(L(nu), e((mu, 1)), 0), (L(mu), e((nu, 1)), 1), (L(nu), z, 1)]))
        if rectangles:
            code.append(
                (TD[mu, nu], -1, 1.0,
                 [(L(nu), e((mu, 1), (nu, -1)), 1), (L(mu), e((nu, -1)), 1), (L(nu), e((nu, -1)), 0)])
            )
    for k, (mu, nu) in enumerate(pairs):
        code.append((0, -1 if k == 0 else 0, wP / 2, [(L(mu), z, 0), (TU[mu, nu], z, 0)]))
        if rectangles:
            code.append((0, 0, wR, [(TD[mu, nu], z, 1), (TU[mu, nu], z, 0)]))
    return code, sorted(TU.values()) + sorted(TD.values())


class staple_stencil_action:
    # S = beta vol (c0 (1 - P) Nd(Nd-1)/2 + c1 (1 - R) Nd(Nd-1))
    #   = beta vol (c0 Nd(Nd-1)/2 + c1 Nd(Nd-1)) - Re sum_x tr O(x)
    # with P, R the plaquette and 2x1 rectangle averages.  The force is the
    # reverse-mode derivative of the stencil; the value needs a single
    # stencil pass.  With rectangles, O is built from the staples as local
    # temporaries (see staple_stencil_code), which the plaquettes and the
    # rectangles share.  Without rectangles nothing is shared, and O is the
    # plain sum of the plaquettes (mu < nu), whose adjoint is one stencil.
    def __init__(self, beta, c0, c1):
        self.beta = beta
        self.c0 = c0
        self.c1 = c1
        self.cache = {}

    def stencil(self, U):
        v = U[0]
        key = (v.grid, v.otype.__name__, len(U))
        if key not in self.cache:
            Nd = len(U)
            N = v.otype.shape[0]
            if self.c1 == 0.0:
                wP = self.beta * self.c0 / N
                code = []
                for mu in range(Nd):
                    for nu in range(mu + 1, Nd):
                        code.append(
                            (0, -1 if len(code) == 0 else 0, wP, g.path().f(mu).f(nu).b(mu).b(nu))
                        )
                self.cache[key] = g.parallel_transport_matrix(U, code, 1)
                return self.cache[key]
            code, temps = staple_stencil_code(
                Nd, self.beta * self.c0 / N, self.beta * self.c1 / N, True
            )
            pts = sorted({(0,) * Nd} | {p for (t, a, w, fl) in code for (f, p, aa) in fl})
            pm = {p: i for i, p in enumerate(pts)}
            ccode = [(t, a, w, [(f, pm[p], aa) for (f, p, aa) in fl]) for (t, a, w, fl) in code]
            # passed fields: 0 = O, 1..Nd = links
            st = g.stencil.matrix(g.lattice(v.grid, v.otype), pts, ccode, temporaries=temps)
            self.cache[key] = st
        return self.cache[key]

    def loops(self, U):
        # the weighted sum of the loops O (plain links or nodes)
        st = self.stencil(U)
        if isinstance(st, g.parallel_transport_matrix):
            return st(U)
        O = U[0].new() if isinstance(U[0], rad.node_base) else g.lattice(U[0])
        st(O, *U)
        return O

    def constant(self, U):
        Nd = len(U)
        vol = U[0].grid.gsites
        return float(self.beta * vol * (self.c0 * Nd * (Nd - 1) / 2 + self.c1 * Nd * (Nd - 1)))

    def __call__(self, aU):
        # plain links: the value; links as nodes: a node
        if isinstance(aU, rad.node_base):
            aU = [aU[mu] for mu in range(len(aU))]
        if not isinstance(aU[0], rad.node_base):
            return self.constant(aU) - g.sum(g.trace(self.loops(aU))).real
        tr = g.sum(g.trace(self.loops(aU)))
        return self.constant(aU) - (tr + g.adj(tr)) * 0.5

    def gradient(self, U, dU):
        # one reusable graph per (grid, otype); the functional replaces the
        # leaf values with U
        v = U[0]
        key = ("functional", v.grid, v.otype.__name__, len(U))
        if key not in self.cache:
            nU = [rad.node(g.copy(u)) for u in U]
            self.cache[key] = self(nU).functional(*nU)
        return self.cache[key].gradient(U, dU)
