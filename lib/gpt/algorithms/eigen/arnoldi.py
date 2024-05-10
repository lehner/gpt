#
#    GPT - Grid Python Toolkit
#    Copyright (C) 2020-22  Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)
#                  2022     Raphael Lehner (raphael.lehner@physik.uni-regensburg.de)
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
from gpt.params import params_convention
import numpy as np
import sys


# Arnoldi iteration
class arnoldi_iteration:
    def __init__(self, mat, src):
        # params
        self.mat = mat

        # verbosity
        self.verbose = g.default.is_verbose("arnoldi")

        # set initial vector
        self.basis = [g.eval(src / g.norm2(src) ** 0.5)]

        # matrix elements
        self.H = []

    def __call__(self, second_orthogonalization=True):
        t0 = g.time()
        new = g.lattice(self.basis[-1])
        self.mat(new, self.basis[-1])
        t1 = g.time()
        ips = np.zeros((len(self.basis) + 1,), np.complex128)
        g.orthogonalize(new, self.basis, ips[0:-1])
        if second_orthogonalization:
            delta_ips = np.zeros((len(self.basis) + 1,), np.complex128)
            g.orthogonalize(new, self.basis, delta_ips[0:-1])
            # the following line may be omitted
            ips += delta_ips
        ips[-1] = g.norm2(new) ** 0.5
        new /= ips[-1]
        self.basis.append(new)
        self.H.append(ips)
        t2 = g.time()

        if self.verbose:
            g.message(
                f"Arnoldi: len(H) = {len(self.H)} took {t1-t0} s for matrix and {t2-t1} s for linear algebra"
            )

    def hessenberg(self):
        n = len(self.H)
        H = np.zeros((n, n), np.complex128)
        for i in range(n - 1):
            H[0 : (i + 2), i] = self.H[i]
        H[:, n - 1] = self.H[n - 1][0:n]
        return H

    def little_eig(self, H):
        t0 = g.time()
        evals, little_evec = np.linalg.eig(H)
        t1 = g.time()

        eps = np.abs([little_evec[:, i][-1] for i in range(len(evals))])

        # sort such that most converged are at end
        idx = (-eps).argsort()

        if self.verbose:
            g.message(f"Arnoldi: eig(H) in {t1-t0} s")

            if any(np.abs(evals) < 1e-14):
                g.message(
                    "Arnoldi: Warning: Some eigenvalues of H are tiny (< 1e-14), this may indicate insufficiently orthogonalized basis vectors"
                )

        return evals[idx], little_evec[:, idx]

    def rotate_basis_to_evec(self, little_evec):
        n = len(self.H)

        t0 = g.time()
        g.rotate(self.basis[0:n], np.ascontiguousarray(little_evec.T), 0, n, 0, n)
        t1 = g.time()

        if self.verbose:
            g.message(f"Arnoldi: rotate in {t1-t0} s")

        return self.basis[0:n]

    def single_evec(self, little_evec, i):
        n = len(self.H)
        test = g.lattice(self.basis[0])
        g.linear_combination(test, self.basis[0:n], little_evec[:, i])
        return test

    def implicit_restart(self, H, evals, p):
        n = len(self.H)
        k = n - p
        Q = np.identity(n, np.complex128)
        eye = np.identity(n, np.complex128)

        t0 = g.time()
        for i in range(p):
            Qi, Ri = np.linalg.qr(H - evals[i] * eye)
            H = Ri @ Qi + evals[i] * eye
            Q = Q @ Qi
        t1 = g.time()

        if self.verbose:
            g.message(f"Arnoldi: QR in {t1-t0} s")

        r = g.eval(self.basis[k] * H[k, k - 1] + self.basis[-1] * self.H[-1][-1] * Q[n - 1, k - 1])
        rn = g.norm2(r) ** 0.5

        t0 = g.time()
        g.rotate(self.basis, np.ascontiguousarray(Q.T), 0, k, 0, n)
        t1 = g.time()

        if self.verbose:
            g.message(f"Arnoldi: rotate in {t1-t0} s")

        self.basis = self.basis[0:k]
        self.basis.append(g.eval(r / rn))
        self.H = [[H[j, i] for j in range(i + 2)] for i in range(k)]
        self.H[-1][-1] = rn


class arnoldi:
    @g.params_convention(
        Nmin=None,
        Nmax=None,
        Nstep=None,
        Nstop=None,
        resid=None,
        implicit_restart=False,
    )
    def __init__(self, params):
        self.params = params
        assert params["Nstop"] <= params["Nmin"]

    # TODO: add checkpointing along lines of irl.py
    def __call__(self, mat, src):
        # verbosity
        self.verbose = g.default.is_verbose("arnoldi")

        # Nstop
        Nstop = self.params["Nstop"]

        # arnoldi base
        a = arnoldi_iteration(mat, src)

        # main loop
        for i in range(self.params["Nmax"]):
            a()

            if i >= self.params["Nmin"] and i % self.params["Nstep"] == 0:
                t0 = g.time()
                H = a.hessenberg()
                t1 = g.time()

                if self.verbose:
                    g.message(f"Arnoldi {i}: hessenberg() in {t1-t0} s")

                evals, little_evec = a.little_eig(H)

                if self.converged(a, mat, evals, little_evec):
                    return a.rotate_basis_to_evec(little_evec)[-Nstop:], evals[-Nstop:]

                if self.params["implicit_restart"]:
                    a.implicit_restart(H, evals, self.params["Nstep"])

        t0 = g.time()
        H = a.hessenberg()
        t1 = g.time()

        if self.verbose:
            g.message(f"Arnoldi: hessenberg() in {t1-t0} s")

        # return results wether converged or not
        evals, little_evec = a.little_eig(H)
        return a.rotate_basis_to_evec(little_evec)[-Nstop:], evals[-Nstop:]

    def converged(self, a, mat, evals, little_evec):
        evals_max = np.max(np.abs(evals))

        Nstop = self.params["Nstop"]
        idx0 = len(evals) - Nstop
        idx1 = len(evals)
        n = 1
        Nconv = 0
        while True:
            idx = idx0 + n - 1
            if idx >= idx1:
                idx = idx1 - 1
            n *= 2

            ev, eps2 = g.algorithms.eigen.evals(mat, [a.single_evec(little_evec, idx)])

            eps2 = eps2[0] / evals_max**2.0

            if self.verbose:
                g.message(f"eval[{idx1 - idx - 1}] = {ev[0]} ; eps^2 = {eps2}")

            if eps2 < self.params["resid"]:
                Nconv = max([Nconv, idx1 - idx])

            if idx == idx1 - 1:
                break

        if self.verbose:
            g.message(f"Arnoldi: {Nconv} eigenmodes converged")

        return Nconv >= Nstop
