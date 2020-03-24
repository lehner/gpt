#
# GPT
#
# Authors: Christoph Lehner 2020
#

import gpt

def orthogonalize(w,basis):
    for v in basis:
        ip=gpt.innerProduct(v,w)
        w -= ip*v

def rotateJ(r,basis,Qt,j,k0,k1,Nm):
    # TODO: use faster Grid version
    r[:]=0
    for k in range(k0,k1):
        r += Qt[j,k] * basis[k]

def rotate(basis,Qt,j0,j1,k0,k1,Nm):
    # TODO: use faster Grid version
    newBasis=[ gpt.lattice(b) for b in basis ]
    for j in range(j0,j1):
        rotateJ(newBasis[j],basis,Qt,j,k0,k1,Nm)
    for i in range(len(newBasis)):
        basis[i] @= newBasis[i]
