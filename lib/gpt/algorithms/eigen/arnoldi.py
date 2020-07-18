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

        new = self.mat(self.basis[-1])
        ips = np.zeros((len(self.basis) + 1,), np.complex128)
        g.orthogonalize(new, self.basis, ips[0:-1])
        ips[-1] = g.norm2(new) ** 0.5
        new /= ips[-1]
        self.basis.append(new)
        self.H.append(ips)

    def hessenberg(self):

        n = len(self.H)
        H = np.zeros((n, n), np.complex128)
        for i in range(n - 1):
            H[0 : (i + 2), i] = self.H[i]
        H[:, n - 1] = self.H[n - 1][0:n]
        return H

    def little_eig(self):

        H = self.hessenberg()
        evals, little_evec = np.linalg.eig(H)
        idx = evals.argsort()
        return evals[idx], little_evec[:, idx]

    def rotate_basis_to_evec(self, little_evec):
        n = len(self.H)
        g.rotate(self.basis[0:n], np.ascontiguousarray(little_evec.T), 0, n, 0, n)
        return self.basis[0:n]

    def single_evec(self, little_evec, i):
        n = len(self.H)
        test = g.lattice(self.basis[0])
        g.linear_combination(test, self.basis[0:n], little_evec[:, i])
        return test
