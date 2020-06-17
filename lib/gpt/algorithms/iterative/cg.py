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

    @g.params_convention(eps = 1e-15, maxiter = 1000000)
    def __init__(self, params):
        self.params = params
        self.eps = params["eps"]
        self.maxiter = params["maxiter"]
        self.history = None
        
    def __call__(self, mat):

        otype,grid,cb=None,None,None
        if type(mat) == g.matrix_operator:
            otype,grid,cb=mat.otype,mat.grid,mat.cb
            mat = mat.mat 
            # remove wrapper for performance benefits

        def inv(psi, src):
            assert(src != psi)
            self.history = []
            verbose=g.default.is_verbose("cg")
            t0=g.time()
            p,mmp,r=g.copy(src),g.copy(src),g.copy(src)
            guess=g.norm2(psi)
            mat(mmp,psi) # in, out
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
                mat(mmp, p)
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
        
        return g.matrix_operator(mat = inv, inv_mat = mat, 
                                 otype = otype, zero = (True,False),
                                 grid = grid, cb = cb)
