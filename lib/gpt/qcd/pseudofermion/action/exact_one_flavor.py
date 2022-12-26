#
#    GPT - Grid Python Toolkit
#    Copyright (C) 2022  Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)
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
from gpt.qcd.pseudofermion.action.base import action_base


Pplus = (g.gamma["I"].tensor() + g.gamma[5].tensor()) * 0.5
Pminus = (g.gamma["I"].tensor() - g.gamma[5].tensor()) * 0.5


class exact_one_flavor_ratio(action_base):
    def __init__(self, fermion, m1, m2, inverter):
        self.m1 = m1
        self.m2 = m2
        super().__init__([fermion(m1, m2), fermion(m1, m1)], inverter, fermion)

    def matrix(self, fields):
        M12, M11, U, phi = self._updated(fields)

        P12 = M12.propagator(self.inverter)
        P11 = M11.propagator(self.inverter)

        m1 = self.m1
        m2 = self.m2

        def _mat(dst, src):
            dst @= (
                src
                + (m2 - m1) * Pplus * g.gamma[5] * P12 * Pplus * src
                + (m1 - m2) * Pminus * g.gamma[5] * P11 * Pminus * src
            )

        return g.matrix_operator(_mat)

    def matrix_spectral_range(self, fields, algorithm):
        evec, evals = algorithm(self.matrix(fields), fields[-1])
        return (min(evals.real), max(evals.real))

    def __call__(self, fields):
        phi = fields[-1]
        M = self.matrix(fields)
        psi = g(M * phi)
        return g.inner_product(phi, psi).real

    def inv_sqrt_matrix(self, fields, rational_function):
        U = fields[0:-1]

        # use woodbury formula, see also https://arxiv.org/pdf/1706.05843.pdf
        #
        # rational(1 + U C^-1 V)
        # = pf0 + pf_i (1 + shift_i + U C^-1 V)^-1
        # = pf0 + pf_i gamma_i (1 - gamma_i U (C + gamma_i V U)^-1 V )
        # with gamma_i = (1 + shift_i)^{-1}
        #

        cs = []
        dm = []
        diag = rational_function.pf0
        for r, p in zip(rational_function.r, rational_function.poles):
            gamma_i = 1.0 / (1.0 - p)
            diag += r * gamma_i
            cs.append(-r * gamma_i**2.0)
            dm.append(gamma_i)

        m1 = self.m1
        m2 = self.m2

        P11 = [
            self.operator(m1, m1 + d * (m2 - m1)).updated(U).propagator(self.inverter) for d in dm
        ]
        P12 = [
            self.operator(m1 + d * (m2 - m1), m2).updated(U).propagator(self.inverter) for d in dm
        ]

        def _mat(dst, src):
            dst @= diag * src
            for i in range(len(cs)):
                dst += cs[i] * (m2 - m1) * Pplus * g.gamma[5] * P12[i] * Pplus * src
                dst -= cs[i] * (m2 - m1) * Pminus * g.gamma[5] * P11[i] * Pminus * src
            dst *= rational_function.norm

        return g.matrix_operator(_mat)

    def draw(self, fields, rng, rational_function):
        phi = fields[-1]

        inv_sqrt_M = self.inv_sqrt_matrix(fields, rational_function)

        eta = g.lattice(phi)
        rng.cnormal(eta, sigma=2.0**-0.5)  # 1/sqrt(2)

        phi @= inv_sqrt_M * eta

        return g.norm2(eta)

    def gradient(self, fields, dfields):
        M12, M11, U, phi = self._updated(fields)

        frc = self._allocate_force(U)

        m1 = self.m1
        m2 = self.m2

        w_plus = g(
            self.inverter(M12.adj())
            * M12.R
            * g.gamma[5]
            * M12.ImportUnphysicalFermion
            * Pplus
            * phi
        )
        w_minus = g(
            self.inverter(M11.adj())
            * M11.R
            * g.gamma[5]
            * M11.ImportUnphysicalFermion
            * Pminus
            * phi
        )

        w2_plus = g(g.gamma[5] * M12.R * M12.Dminus.adj() * w_plus)
        w3_plus = g(Pplus * phi)

        w2_minus = g(g.gamma[5] * M11.R * M11.Dminus.adj() * w_minus)
        w3_minus = g(Pminus * phi)

        self._accumulate(frc, M12.M_projected_gradient(w_plus, w2_plus), m1 - m2)
        self._accumulate(
            frc,
            M12.ImportPhysicalFermionSource_projected_gradient(w_plus, w3_plus),
            m2 - m1,
        )
        self._accumulate(frc, M11.M_projected_gradient(w_minus, w2_minus), m2 - m1)
        self._accumulate(
            frc,
            M11.ImportPhysicalFermionSource_projected_gradient(w_minus, w3_minus),
            m1 - m2,
        )

        dS = []
        for f in dfields:
            mu = fields.index(f)
            if mu < len(fields) - 1:
                dS.append(g.qcd.gauge.project.traceless_hermitian(frc[mu]))
            else:
                raise Exception("not implemented")

        return dS
