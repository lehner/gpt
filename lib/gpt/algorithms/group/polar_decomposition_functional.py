#
#    GPT - Grid Python Toolkit
#    Copyright (C) 2024  Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)
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

from gpt.core.group import differentiable_functional


def decompose(w, phase0=None):
    if phase0 is None:
        phase0 = g.complex(w.grid)
        phase0[:] = 0

    h, u = g.matrix.polar.decompose(w)
    rel_det = g(g.matrix.det(g.component.exp(-1j * phase0) * u))
    rel_phase = g(g.component.log(rel_det) / 1j / u.otype.Ndim)
    phase = g(phase0 + rel_phase)
    su = g(g.component.exp(-1j * phase) * u)
    su.otype = g.ot_matrix_su_n_fundamental_group(u.otype.Ndim)
    return h, phase, su


class polar_decomposition_functional(differentiable_functional):
    def __init__(self, u_functional, h_functional):
        self.u_functional = u_functional
        self.h_functional = h_functional
        self.reference_phases = None

    def reduce(self, fields):
        h = []
        p = []
        u = []
        p0 = self.reference_phases if self.reference_phases is not None else [None] * len(fields)
        for i in range(len(fields)):
            hf, pf, uf = decompose(fields[i], p0[i])
            h.append(hf)
            p.append(pf)
            u.append(uf)
        self.reference_phases = p
        return h, p, u

    def __call__(self, fields):
        h, p, u = self.reduce(fields)
        return self.u_functional(u) + self.h_functional(h)

    def gradient(self, fields, dfields):
        indices = [fields.index(f) for f in dfields]

        h, p, u = self.reduce(fields)

        u_gradient = self.u_functional.gradient(u, [u[mu] for mu in indices])
        h_gradient = self.h_functional.gradient(h, [h[mu] for mu in indices])

        dSdA = []

        grid = fields[0].grid
        lsites = grid.gsites // grid.Nprocessors
        d_cartesian_space = g.group.cartesian(fields[0])
        cartesian_space = g.group.cartesian(u[0])
        gen = cartesian_space.otype.generators(grid.precision.complex_dtype)
        for nu, mu in enumerate(indices):
            # fill jacobian
            Nc = fields[0].otype.Ndim
            N = Nc**2 * 2
            Na = len(gen)
            jac = np.ndarray(shape=(lsites, N, N), dtype=np.float64)
            for a in range(Na):
                ta = gen[a]
                ta_u = g(ta * u[mu])
                h_ta_u = g(h[mu] * ta_u)
                eitheta_h_ta_u = g(g.component.exp(1j * p[mu]) * h_ta_u)
                eitheta_ta_u = g(g.component.exp(1j * p[mu]) * ta_u)

                v_eitheta_h_ta_u = eitheta_h_ta_u[:]
                v_eitheta_ta_u = eitheta_ta_u[:]
                for i in range(Nc):
                    for j in range(Nc):
                        jac[:, 0 * Na + a, 0 * Nc * Nc + i * Nc + j] = -v_eitheta_h_ta_u[
                            :, i, j
                        ].imag
                        jac[:, 0 * Na + a, 1 * Nc * Nc + i * Nc + j] = v_eitheta_h_ta_u[
                            :, i, j
                        ].real
                        jac[:, 1 * Na + a, 0 * Nc * Nc + i * Nc + j] = v_eitheta_ta_u[:, i, j].real
                        jac[:, 1 * Na + a, 1 * Nc * Nc + i * Nc + j] = v_eitheta_ta_u[:, i, j].imag

            v_w = fields[mu][:]
            eitheta_u = g(g.component.exp(1j * p[mu]) * u[mu])
            v_eitheta_u = eitheta_u[:]
            for i in range(Nc):
                for j in range(Nc):
                    jac[:, 2 * Na + 0, 0 * Nc * Nc + i * Nc + j] = -v_w[:, i, j].imag
                    jac[:, 2 * Na + 0, 1 * Nc * Nc + i * Nc + j] = v_w[:, i, j].real
                    jac[:, 2 * Na + 1, 0 * Nc * Nc + i * Nc + j] = v_eitheta_u[:, i, j].real
                    jac[:, 2 * Na + 1, 1 * Nc * Nc + i * Nc + j] = v_eitheta_u[:, i, j].imag

            inv_jac = np.linalg.inv(jac)

            # next, project out each a
            gr_w = np.zeros(shape=(lsites, 2 * Nc * Nc), dtype=np.complex128)
            for a in range(Na):
                u_gradient_a = g(g.trace(gen[a] * u_gradient[nu]))
                v_u_gradient_a = u_gradient_a[:]

                h_gradient_a = g(g.trace(gen[a] * h_gradient[nu]))
                v_h_gradient_a = h_gradient_a[:]

                gr_w += inv_jac[:, :, 0 * Na + a] * v_u_gradient_a.real
                gr_w += inv_jac[:, :, 1 * Na + a] * v_h_gradient_a.real / 2.0

            h_gradient_a = g(g.trace(h_gradient[nu]))
            v_h_gradient_a = h_gradient_a[:]
            gr_w += inv_jac[:, :, 2 * Na + 1] * v_h_gradient_a.real / 2.0

            x = gr_w[:, 0 : Nc * Nc].reshape(lsites, Nc, Nc)
            x += 1j * gr_w[:, Nc * Nc : 2 * Nc * Nc].reshape(lsites, Nc, Nc)
            y = g.lattice(d_cartesian_space)
            y[:] = np.ascontiguousarray(x)
            dSdA.append(g(2 * y))

        return dSdA
