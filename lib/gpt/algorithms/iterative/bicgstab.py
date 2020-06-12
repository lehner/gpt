#
#    GPT - Grid Python Toolkit
#    Copyright (C) 2020  Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)
#                  2020  Daniel Richtmann
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
from time import time

class bicgstab:

    @g.params_convention(eps = 1e-15, maxiter = 1000000)
    def __init__(self, params):
        self.params = params
        self.eps = params["eps"]
        self.maxiter = params["maxiter"]

    def __call__(self, mat):

        def inv(psi, src):
            verbose=g.default.is_verbose("bicgstab")
            t0 = time()

            r, rhat, p, s = g.copy(src), g.copy(src), g.copy(src), g.copy(src)
            mmpsi, mmp, mms = g.copy(src), g.copy(src), g.copy(src)

            rho, rhoprev, alpha, omega = 1., 1., 1., 1.

            mat(mmpsi, psi)
            r @= src - mmpsi

            rhat @= r
            p @= r
            mmp @= r

            ssq = g.norm2(src)
            rsq = self.eps**2. * ssq

            for k in range(self.maxiter):
                rhoprev = rho
                rho = g.innerProduct(rhat, r).real

                beta = (rho / rhoprev) * (alpha / omega)

                p @= r + beta * p - beta * omega * mmp

                mat(mmp, p)
                alpha = rho / g.innerProduct(rhat, mmp).real

                s @= r - alpha * mmp

                mat(mms, s)
                ip, mms2 = g.innerProductNorm2(mms, s)

                if mms2 == 0.:
                    continue

                omega = ip.real / mms2

                psi += alpha * p + omega * s

                r2 = g.axpy_norm2(r, -omega, mms, s)

                if verbose:
                    g.message("res^2[ %d ] = %g" % (k, r2))

                if r2 <= rsq:
                    if verbose:
                        t1 = time()
                        g.message("Converged in %g s" % (t1 - t0))
                    break
        
        otype = None
        grid = None
        if type(mat) == g.matrix_operator:
            otype = mat.otype
            grid = mat.grid

        return g.matrix_operator(mat = inv, inv_mat = mat, 
                                 otype = otype, zero = (True,False),
                                 grid = grid)
