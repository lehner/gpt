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

class cg:

    def __init__(self, params):
        self.params = params
        self.eps = params["eps"]
        self.maxiter = params["maxiter"]
        self.history = None

    def __call__(self, mat, src, psi):
        assert(src != psi)
        self.history = []
        verbose=g.default.is_verbose("cg")
        t0=g.time()
        p,mmp,r=g.copy(src),g.copy(src),g.copy(src)
        guess=g.norm2(psi)
        mat(psi,mmp) # in, out
        d=g.innerProduct(psi,mmp).real
        b=g.norm2(mmp)
        r @= src - mmp
        p @= r
        a = g.norm2(p)
        cp = a
        ssq = g.norm2(src)
        rsq = self.eps**2. * ssq
        for k in range(1,self.maxiter+1):
            c=cp
            mat(p, mmp)
            dc=g.innerProduct(p,mmp)
            d=dc.real
            a = c / d
            cp=g.axpy_norm2(r, -a, mmp, r)
            b = cp / c
            psi += a*p
            p @= b*p+r
            self.history.append(cp)
            if verbose:
                g.message("res^2[ %d ] = %g" % (k,cp))
            if cp <= rsq:
                if verbose:
                    t1=g.time()
                    g.message("Converged in %g s" % (t1-t0))
                break
