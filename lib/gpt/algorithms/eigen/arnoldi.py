#
#    GPT - Grid Python Toolkit
#    Copyright (C) 2020  Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)
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

    def little_eig(self):

        t0 = g.time()
        H = self.hessenberg()
        t1 = g.time()
        evals, little_evec = np.linalg.eig(H)
        t2 = g.time()
        idx = evals.argsort()

        if self.verbose:
            g.message(f"Arnoldi: hessenberg() in {t1-t0} s and eig(H) in {t2-t1} s")

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


class arnoldi:
    @params_convention(Nmin=None, Nmax=None, Nstep=None, Nstop=None, resid=None)
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
                evals, little_evec = a.little_eig()
                if self.converged(a, mat, evals, little_evec):
                    return a.rotate_basis_to_evec(little_evec)[-Nstop:], evals[-Nstop:]

        # return results wether converged or not
        evals, little_evec = a.little_eig()
        return a.rotate_basis_to_evec(little_evec)[-Nstop:], evals[-Nstop:]

    def converged(self, a, mat, evals, little_evec):

        n = 1
        Nconv = 0
        while True:
            idx = len(evals) - n
            n *= 2
            if idx < 0:
                idx = 0

            try:
                g.algorithms.eigen.evals(
                    mat,
                    [a.single_evec(little_evec, idx)],
                    check_eps2=evals[-1] ** 2.0 * self.params["resid"],
                )
            except g.algorithms.eigen.EvalsNotConverged:
                break

            Nconv = len(evals) - idx

            if idx == 0:
                break

        if self.verbose:
            g.message(f"Arnoldi: {Nconv} eigenmodes converged")

        return Nconv >= self.params["Nstop"]
