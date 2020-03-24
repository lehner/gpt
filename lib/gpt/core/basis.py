#
# GPT
#
# Authors: Christoph Lehner 2020
#

import gpt
import cgpt

def orthogonalize(w,basis):
    for v in basis:
        ip=gpt.innerProduct(v,w)
        w -= ip*v

def linear_combination(r,basis,Qt):
    return cgpt.linear_combination(r.obj,basis,Qt)

def rotate(basis,Qt,j0,j1,k0,k1):
    return cgpt.rotate(basis,Qt,j0,j1,k0,k1)

def qr_decomp(lmd,lme,Nk,Nm,Qt,Dsh,kmin,kmax):
    return cgpt.qr_decomp(lmd,lme,Nk,Nm,Qt,Dsh,kmin,kmax)
