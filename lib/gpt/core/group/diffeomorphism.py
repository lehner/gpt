#
#    GPT - Grid Python Toolkit
#    Copyright (C) 2022-24  Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)
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
import sys


class diffeomorphism:
    def __init__(self):
        return

    def __call__(self, fields):
        raise NotImplementedError()

    def jacobian(self, fields, fields_prime, dfields):
        raise NotImplementedError()


class local_diffeomorphism(diffeomorphism):
    def __init__(self):
        return

    def __call__(self, fields):
        raise NotImplementedError()

    def log_det_jacobian(self, fields):
        raise NotImplementedError()

    def approximate_jacobian(self, fields, eps, one):
        fields_prime_orig = self(fields)

        grid = fields[0].grid
        dt = grid.precision.complex_dtype
        otype = fields[0].otype
        cartesian_otype = otype.cartesian()
        generators = cartesian_otype.generators(dt)
        ng = len(generators)

        nd = len(fields)
        dA = g.lattice(grid, cartesian_otype)
        dAprime = g.lattice(grid, cartesian_otype)
        dAprimen = g.lattice(grid, cartesian_otype)
        V = g.lattice(grid, otype)

        nlc = np.prod(grid.ldimensions)

        # create local nd x nd x ng x ng matrix
        Jac = np.zeros(shape=(nlc, nd, nd, ng, ng), dtype=dt)

        nd = len(fields)
        for fa in range(nd):
            for a in range(ng):
                # generate one-form dU_a = tr(dU U^-1 Ta)
                dA @= one * generators[a] * eps

                g.convert(V, dA)

                fields_shift = [
                    g(g.group.compose(V, fields[i])) if i == fa else fields[i] for i in range(nd)
                ]
                fields_prime_shift = self(fields_shift)

                fields_shift_n = [
                    g(g.group.compose(g.group.inverse(V), fields[i])) if i == fa else fields[i]
                    for i in range(nd)
                ]
                fields_prime_shift_n = self(fields_shift_n)

                for fb in range(nd):
                    Vprime = g.group.compose(
                        fields_prime_shift[fb], g.group.inverse(fields_prime_orig[fb])
                    )
                    g.convert(dAprime, Vprime)

                    Vprime = g.group.compose(
                        fields_prime_shift_n[fb], g.group.inverse(fields_prime_orig[fb])
                    )
                    g.convert(dAprimen, Vprime)

                    dAprime @= 0.5 * dAprime - 0.5 * dAprimen

                    Jba = cartesian_otype.coordinates(dAprime)
                    for b in range(ng):
                        Jac[:, fa, fb, a, b] = Jba[b][:].flatten() / eps

        return Jac

    def assert_log_det_jacobian(self, fields, eps, origin, eps_test):
        # first test locality
        grid = fields[0].grid
        one = g.complex(grid)
        one[:] = 0
        one[origin] = 1

        t = g.timer("assert_log_det_jacobian")

        t("evaluate")
        log_det = self.log_det_jacobian(fields)[origin]

        t("estimate")

        coor = g.coordinates(one)
        coor_idx = grid.lexicographic_index(coor)
        origin_idx = grid.lexicographic_index(np.array([origin]))
        indices = np.arange(len(coor))
        iorigin = indices[coor_idx == origin_idx]
        assert len(iorigin) in [0, 1]

        Jac = self.approximate_jacobian(fields, eps, one)
        nm = Jac.shape[1] * Jac.shape[3]
        Jac_flat = Jac.swapaxes(3, 2).reshape((Jac.shape[0], nm, nm))

        if len(iorigin) == 1:
            iorigin = iorigin[0]
            log_det_appx = float(np.log(np.linalg.det(Jac_flat[iorigin])).real)
        else:
            log_det_appx = 0.0
        log_det_appx = grid.globalsum(log_det_appx)

        t()

        # g.message(t)

        err = abs(log_det - log_det_appx)
        g.message(f"assert_log_det_jacobian: {err} < {eps_test} ; log_det = {log_det} , log_det_appx = {log_det_appx}")
        assert err <= eps_test
