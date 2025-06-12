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

# Ansatz:
# W = H U
# U = e^{i \theta T} SU
# det(U) = e^{i\theta},   \theta \in [0,2\pi]   e^{i \theta T}

# T^2 = T, tr(T) = 1
# e^{i \theta T} = 1 + i \theta T - 1/2 \theta^2 T + ... = (e^{i\theta} - 1) T + 1 = (1-T) + T e^{i\theta}

# \Theta = dU U^{-1} = i T_a \Theta_a  ->  dU = i T_a \Theta_a U
# H = T_a \phi_a + \phi_0
# tr(T_a T_b) = 1/2 \delta_{ab}

# dW = d\phi_a T_a e^{i \theta T} U + d\phi_0 e^{i \theta T} U + i H d\theta T e^{i\theta T} U + i H e^{i \theta T} T_a U \Theta_a = e_{ij} dx_{ij} + i e_{ij} dy_{ij}

# The invariant length element is therefore:
#
# ds^2 = tr[dW dW^\dagger]
#      = + tr[H e^{i\theta T} T_a T_b e^{-i\theta T} H^\dagger] \Theta_a \Theta_b
#        + ...
# Important point is that since dW = dA U and dA is free of U, we always find tr[dW dW^\dagger] to be independent of U and therefore the
# metric tensor and the measure is invariant of U !

# with matrices (e_{ij})_{ab} = \delta_{ia} \delta_{jb}


def embedded_phase(u, phase):
    ep = g.identity(u)

    # TODO: generalize to n dimensions
    ep[:, :, :, :, 0, 0] = phase[:]

    return ep


def decompose(w):
    h, u = g.matrix.polar.decompose(w)

    # u = e^{i \theta T_first_row} su
    # (T_first_row)_{ij} = \delta_{i0} \delta_{j0}  ->  det(u) = e^{i theta tr(T_first_row)} = e^{i theta}
    # need tr(T) = 1 but also e^{i 2pi T} = 1 for continuous behavior
    det = g(g.matrix.det(u))
    theta = g(g.component.log(det) / 1j)
    phase = g(g.component.exp(-1j * theta))
    su = g(embedded_phase(u, phase) * u)
    su.otype = g.ot_matrix_su_n_fundamental_group(u.otype.Ndim)
    return h, theta, su


class polar_decomposition_functional(differentiable_functional):
    def __init__(self, u_functional, h_functional):
        self.u_functional = u_functional
        self.h_functional = h_functional

    def reduce(self, fields):
        h = []
        p = []
        u = []
        for i in range(len(fields)):
            hf, pf, uf = decompose(fields[i])
            h.append(hf)
            p.append(pf)
            u.append(uf)
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
            ep = embedded_phase(u[mu], g.component.exp(1j * p[mu]))

            # dW =
            # + d\phi_a v_eitheta_ta_u
            # + d\phi_0 v_eitheta_u
            # + i d\theta v_w
            # + i v_eitheta_h_ta_u \Theta_a
            # = e_{ij} dx_{ij} + i e_{ij} dy_{ij}

            for a in range(Na):
                ta = gen[a]
                v_eitheta_h_ta_u = g(h[mu] * ep * ta * u[mu])[:]
                v_eitheta_ta_u = g(ta * ep * u[mu])[:]
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

            That = g.mcolor([[1 if i == 0 and j == 0 else 0 for i in range(Nc)] for j in range(Nc)])
            v_w = g(h[mu] * That * ep * u[mu])[:]
            v_eitheta_u = g(ep * u[mu])[:]
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
