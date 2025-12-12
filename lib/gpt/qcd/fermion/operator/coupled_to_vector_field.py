#
#    GPT - Grid Python Toolkit
#    Copyright (C) 2025  Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)
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


def extend(U):

    nd = U[0].grid.nd

    def apply(me):
        if "e" not in me.params:
            return

        e = me.params["e"]
        R = me.ExportPhysicalFermionSolution
        P1 = me.Dminus.adj()
        if me.daggered:
            R = R.adj()
            P1 = P1.adj()

        def MDeriv(dst, left, right):
            right = g(R * right)
            left = g(g.gamma[0] * R * g.gamma[0] * P1 * left)
            for mu in range(nd):
                dst[mu] @= e * g.adj(left) * g.gamma[mu] * right

        def MDerivDag(dst, left, right):
            left = g(R * left)
            right = g(g.gamma[0] * R * g.gamma[0] * P1 * right)
            for mu in range(nd):
                dst[mu] @= -e * g.adj(left) * g.gamma[mu] * right

        # def CheckerboardWrapper(fnc):

        #     def wrapped(dst_cb, left_cb, right_cb):
        #         right = g.lattice(F_grid, otype)
        #         left = g.lattice(F_grid, otype)
        #         right[:] = 0
        #         left[:] = 0
        #         g.set_checkerboard(left, left_cb)
        #         g.set_checkerboard(right, right_cb)
        #         dst = [g.lattice(U_grid, x.otype) for x in dst_cb]
        #         for x in dst:
        #             x[:] = 0
        #         fnc(dst, left, right)
        #         for mu in range(nd):
        #             print(mu, dst_cb[mu].checkerboard(), left_cb.checkerboard(), right_cb.checkerboard())
        #             g.pick_checkerboard(dst_cb[mu].checkerboard(), dst_cb[mu], dst[mu])

        #     return wrapped

        if len(U) == nd * 2:
            # extend to include A fields
            def ext(fnc, fnc2=None):
                def lam(dst, left, right):
                    fnc(dst[0:nd], left, right)
                    if fnc2 is None:
                        for x in dst[nd:]:  # while debugging, make it reproduce
                            x[:] = 1
                    else:
                        fnc2(dst[nd:], left, right)

                return lam

            me._MeoDeriv = ext(me._MeoDeriv)  # , CheckerboardWrapper(MDeriv))
            me._MoeDeriv = ext(me._MoeDeriv)  # , CheckerboardWrapper(MDeriv))
            me._MeoDerivDag = ext(me._MeoDerivDag)  # , CheckerboardWrapper(MDerivDag))
            me._MoeDerivDag = ext(me._MoeDerivDag)  # , CheckerboardWrapper(MDerivDag))
            me._MDeriv = ext(me._MDeriv, MDeriv)
            me._MDerivDag = ext(me._MDerivDag, MDerivDag)
            me._DhopDeriv = ext(me._DhopDeriv)
            me._DhopDerivDag = ext(me._DhopDerivDag)

    return apply
