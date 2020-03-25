#
# GPT
#
# Authors: Christoph Lehner 2020
#
import gpt as g
from time import time

class cg:

    def __init__(self, params):
        self.params = params
        self.eps = params["eps"]
        self.maxiter = params["maxiter"]

    def __call__(self, mat, src, psi):
        verbose=g.default.is_verbose("cg")
        t0=time()
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
            cp=g.axpy_norm(r, -a, mmp, r)
            b = cp / c
            psi += a*p
            p @= b*p+r
            if verbose:
                g.message("res^2[ %d ] = %g" % (k,cp))
            if cp <= rsq:
                if verbose:
                    t1=time()
                    g.message("Converged in %g s" % (t1-t0))
                break
