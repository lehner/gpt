#
#    GPT - Grid Python Toolkit
#    Copyright (C) 2020  Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)
#                  2022  Raphael Lehner (raphael.lehner@physik.uni-regensburg.de)
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

    def __call__(self):

        t0 = g.time()
        new = g.lattice(self.basis[-1])
        self.mat(new, self.basis[-1])
        t1 = g.time()
        ips = np.zeros((len(self.basis) + 1,), np.complex128)
        g.orthogonalize(new, self.basis, ips[0:-1])
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
        idx = evals.argsort()
        # # find z0 for better convergence
        # idx = (np.abs(evals - z0)).argsort()

        if self.verbose:
            g.message(f"Arnoldi: eig(H) in {t1-t0} s")

        return evals[idx], little_evec[:, idx]

    def rotate_basis_to_evec(self, little_evec):
        n = len(self.H)

        t0 = g.time()
        g.rotate(self.basis[0:n], np.ascontiguousarray(little_evec.T), 0, n, 0, n)
        t1 = g.time()

        if self.verbose:
            g.message(f"Arnoldi: rotate in {t1-t0} s")

        return self.basis[0:n]

    def restart(self, H, evals, p, rc, skip, nblock):

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

        r = g.eval(
            self.basis[k] * H[k, k - 1]
            + self.basis[-1] * self.H[-1][-1] * Q[n - 1, k - 1]
        )
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

        if rc % skip == 0:

            t0 = g.time()
            g.orthonormalize(self.basis, nblock)
            t1 = g.time()

            if self.verbose:
                g.message(f"Arnoldi: orthonormalize in {t1-t0} s")


class arnoldi:
    @g.params_convention(
        Nmin=None,
        Nmax=None,
        Nstep=None,
        Nstop=None,
        resid=None,
        restart=False,
        orthonormalize_skip=10,
        orthonormalize_nblock=4,
    )
    def __init__(self, params):
        self.params = params
        assert params["Nstop"] <= params["Nmin"]

    # TODO: add checkpointing along lines of irl.py
    def __call__(self, mat, src):

        # verbosity
        self.verbose = g.default.is_verbose("arnoldi")

        # squared residual
        rsq = g.norm2(src) * self.params["resid"] ** 2.0

        # Nstop
        Nstop = self.params["Nstop"]

        # arnoldi base
        a = arnoldi_iteration(mat, src)

        # restart count
        rc = 0

        # main loop
        for i in range(self.params["Nmax"]):
            a()

            if i >= self.params["Nmin"] and i % self.params["Nstep"] == 0:

                t0 = g.time()
                H = a.hessenberg()
                t1 = g.time()

                if self.verbose:
                    g.message(f"Arnoldi: hessenberg() in {t1-t0} s")

                evals, little_evec = a.little_eig(H)

                if self.converged(a, mat, evals, little_evec, rsq):
                    return a.rotate_basis_to_evec(little_evec)[-Nstop:], evals[-Nstop:]

                if self.params["restart"]:
                    rc += 1
                    a.restart(
                        H,
                        evals,
                        self.params["Nstep"],
                        rc,
                        self.params["orthonormalize_skip"],
                        self.params["orthonormalize_nblock"],
                    )

        t0 = g.time()
        H = a.hessenberg()
        t1 = g.time()

        if self.verbose:
            g.message(f"Arnoldi: hessenberg() in {t1-t0} s")

        # return results wether converged or not
        evals, little_evec = a.little_eig(H)
        return a.rotate_basis_to_evec(little_evec)[-Nstop:], evals[-Nstop:]

    def converged(self, a, mat, evals, little_evec, rsq):

        n = 1
        Nconv = 0
        while True:
            idx = len(evals) - n
            n *= 2
            if idx < 0:
                idx = 0

            if np.abs(a.H[-1][-1] * little_evec[:,idx][-1]) ** 2.0 > rsq:
                break

            Nconv = len(evals) - idx

            if idx == 0:
                break

        if self.verbose:
            g.message(f"Arnoldi: {Nconv} eigenmodes converged")

        return Nconv >= self.params["Nstop"]
