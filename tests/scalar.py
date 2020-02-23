#!/usr/bin/env python3
#
# Authors: Christoph Lehner 2020
#
# Desc.: Illustrate core concepts and features
#
import gpt as g

grid=g.grid(g.default.grid, g.single)

src=g.complex(grid)
src[:]=0
src[0,0,0,0]=1

# Create a free Klein-Gordon operator
def A(dst,src,mass):
    g.eval(dst,mass*src)
    for i in range(4):
        g.eval(dst, dst + g.cshift(src, i, 1) + g.cshift(src, i, -1) - 2*src )

# find largest eigenvalue
dst,tmp=g.lattice(src),g.copy(src)
for it in range(50):
    tmp = g.eval( (1.0/g.norm2(tmp)**0.5) * tmp )
    A(dst,tmp,0.0)
    g.message("Iteration %d %g" % (it,g.norm2(dst)**0.5))
    tmp=dst

# CG
def CG(mat,src,psi,tol,maxit):
    p,mmp,r=g.copy(src),g.copy(src),g.copy(src)
    guess=g.norm2(psi)
    mat(psi,mmp) # in, out
    d=g.innerProduct(psi,mmp).real
    b=g.norm2(mmp)
    g.eval(r,src - mmp)
    g.copy(p,r)
    a = g.norm2(p)
    cp = a
    ssq = g.norm2(src)
    rsq = tol * tol * ssq
    for k in range(1,maxit+1):
        c=cp
        mat(p, mmp)
        dc=g.innerProduct(p,mmp)
        d=dc.real
        a = c / d
        cp=g.axpy_norm(r, -a, mmp, r)
        b = cp / c
        g.eval(psi,a*p+psi)
        g.eval(p,b*p+r)
        g.message("Iter %d -> %g" % (k,cp))
        if cp <= rsq:
            g.message("Converged")
            break

# Perform CG
psi=g.lattice(src)
psi[:]=0
CG(lambda i,o: A(o,i,1.0),src,psi,1e-8,1000)

# Test CG
A(tmp,psi,1.0)
g.message("True residuum:", g.norm2(g.eval(tmp - src)))

