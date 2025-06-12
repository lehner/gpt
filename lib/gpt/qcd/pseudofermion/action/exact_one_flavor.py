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
        M12 = fermion(m1, m2)
        M11 = fermion(m1, m1)
        M12_adj = M12.adj()
        M11_adj = M11.adj()
        super().__init__([M12, M11, M12_adj, M11_adj], inverter, fermion)

    def matrix(self, fields):
        M12, M11, M12_adj, M11_adj, U, phi = self._updated(fields)

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
        self._suspend()
        return (min(evals.real), max(evals.real))

    def __call__(self, fields):
        phi = fields[-1]
        M = self.matrix(fields)
        psi = g(M * phi)
        self._suspend()
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

        operator = self.operator
        inverter = self.inverter

        def _mat(dst, src):
            dst @= diag * src
            for i in range(len(cs)):

                op = operator(m1 + dm[i] * (m2 - m1), m2)
                op.update(U)
                P12_i = op.propagator(inverter)

                dst += cs[i] * (m2 - m1) * Pplus * g.gamma[5] * P12_i * Pplus * src

                P12_i = None

                op = operator(m1, m1 + dm[i] * (m2 - m1))
                op.update(U)
                P11_i = op.propagator(inverter)

                dst -= cs[i] * (m2 - m1) * Pminus * g.gamma[5] * P11_i * Pminus * src

                P11_i = None
                op = None

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
        M12, M11, M12_adj, M11_adj, U, phi = self._updated(fields)

        frc = self._allocate_force(U)

        # g.barrier()
        # g.message("checkmark 0")
        # g.barrier()

        inv_M12_adj = self.inverter(M12_adj)

        # g.barrier()
        # g.message("checkmark 1")
        # g.barrier()

        inv_M11_adj = self.inverter(M11_adj)

        # g.barrier()
        # g.message("checkmark 2")
        # g.barrier()

        m1 = self.m1
        m2 = self.m2

        w_plus = g(inv_M12_adj * M12.R * g.gamma[5] * M12.ImportUnphysicalFermion * Pplus * phi)

        # g.barrier()
        # g.message("checkmark 3")
        # g.barrier()

        w_minus = g(inv_M11_adj * M11.R * g.gamma[5] * M11.ImportUnphysicalFermion * Pminus * phi)

        # g.barrier()
        # g.message("checkmark 4")
        # g.barrier()

        w2_plus = g(g.gamma[5] * M12.R * M12.Dminus.adj() * w_plus)
        w3_plus = g(Pplus * phi)

        w2_minus = g(g.gamma[5] * M11.R * M11.Dminus.adj() * w_minus)
        w3_minus = g(Pminus * phi)

        # g.barrier()
        # g.message("checkmark 5")
        # g.barrier()

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
                # not yet implemented
                dS.append(None)

        self._suspend()
        return dS
