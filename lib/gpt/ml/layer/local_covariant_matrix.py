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
import numpy as np
from gpt.ml.function import function


class local_covariant_matrix(function):
    """A residual block on C channels X_c of N x N matrix fields that
    transform as X_c(x) -> V(x) X_c(x) V(x)^dag:

      Y_c = sum_d (a_cd X_d + b_cd X_d^dag) + beta_c 1
      Z_c = Y_c Y_{c+1 mod C}
      Z_c <- relu(alpha_c (q_c - mean_c) inv_std_c + gamma_c) Z_c   (gate=True)
      X_c <- X_c + gain_c Z_c

    with the invariant q_c = tr(Y_c Y_c^dag) / N.  Every operation is
    covariant (scalar coefficients, products, adjoints, the identity, and
    gates that are functions of invariants); relu acts on complex numbers as
    z for Re z > 0 and 0 otherwise.  Input and output: X (a list of C
    fields).  Parameters (complex numbers): a, b (C x C, row-major), beta,
    gain, and with the gate alpha, gamma.

    The block is the identity for gain = 0.  initialize draws gain at the
    distance scale from the identity (scale = 0: exactly the identity, with
    nonzero gradients w.r.t. gain since Z != 0) and the inner weights at
    O(1) / sqrt(2 C + 1) (the number of terms of Y_c).  Constants: mean, inv_std (the gate
    references, set by calibrate to the mean and inverse standard deviation
    of q_c over the sites of the calibration data, so that the gate argument
    varies by O(alpha); frozen to keep the block site-local)."""

    def __init__(self, template, n_channels, gate=False, scale=0.1):
        C = n_channels
        self.C, self.gate, self.scale = C, gate, scale
        self.N = template.otype.shape[0]
        # the unit matrix and the unit complex field (no slots: they follow
        # from the template)
        self.identity = g.identity(template)
        self.one = g.complex(template.grid)
        self.one[:] = 1
        parameters = [("a", [0j] * (C * C)), ("b", [0j] * (C * C)), ("beta", [0j] * C), ("gain", [0j] * C)]
        constants = []
        if gate:
            parameters += [("alpha", [0j] * C), ("gamma", [0j] * C)]
            constants = [("mean", [0.0] * C), ("inv_std", [1.0] * C)]
        X = [template] * C
        super().__init__([("X", X)], [("X", X)], parameters, constants)

    def initialize(self, rng, scale=None):
        # inner weights O(1) / sqrt(fan-in); the output gain at the distance
        # scale from the identity; gates start close to 1, away from the kink
        C = self.C
        scale = self.scale if scale is None else scale

        def r(n, sigma, shift=0.0):
            return [shift + sigma * rng.normal_element(0j) / np.sqrt(2) for _ in range(n)]

        fan_in = np.sqrt(2 * C + 1)
        self["a"], self["b"], self["beta"] = r(C * C, 1 / fan_in), r(C * C, 1 / fan_in), r(C, 1 / fan_in)
        self["gain"] = r(C, scale)
        if self.gate:
            self["alpha"], self["gamma"] = r(C, 0.1), r(C, 0.1, 1.0)

    def _mix(self, X, a, b, beta):
        C = self.C
        Xa = [g.adj(x) for x in X]
        return [
            sum(a[c * C + d] * X[d] + b[c * C + d] * Xa[d] for d in range(C)) + beta[c] * self.identity
            for c in range(C)
        ]

    def _invariants(self, Y):
        return [g(g.trace(y * g.adj(y)) * (1.0 / self.N)) for y in Y]

    def calibrate(self, samples):
        # the gate references from the invariants of the calibration samples
        # (plain inputs) at the current weights
        if not self.gate:
            return
        a, b, beta = self._parameters.group(self.parameters())[0:3]
        q = [self._invariants(self._mix(X, a, b, beta)) for (X,) in samples]
        mean, inv_std = [], []
        for c in range(self.C):
            qc = [x[c] for x in q]
            n = sum(x.grid.gsites for x in qc)
            m = sum(g.sum(x).real for x in qc) / n
            m2 = sum(g.sum(g(x * x)).real for x in qc) / n
            mean.append(m)
            inv_std.append(1.0 / np.sqrt(m2 - m**2))
        self["mean"], self["inv_std"] = mean, inv_std

    def evaluate(self, inputs, parameters, constants):
        (X,) = inputs
        a, b, beta = parameters[0:3]
        Y = self._mix(X, a, b, beta)
        Z = [Y[c] * Y[(c + 1) % self.C] for c in range(self.C)]
        if self.gate:
            alpha, gamma = parameters[4:6]
            mean, inv_std = constants
            for c, q in enumerate(self._invariants(Y)):
                # alpha (q - mean) inv_std + gamma
                s = g(alpha[c] * q * inv_std[c] + (gamma[c] - alpha[c] * mean[c] * inv_std[c]) * self.one)
                Z[c] = g.component.relu()(s) * Z[c]
        gain = parameters[3]
        return [[g(x + gc * z) for x, gc, z in zip(X, gain, Z)]]
