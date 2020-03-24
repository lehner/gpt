#
# GPT
#
# Authors: Christoph Lehner 2020
#
#          The history of this version of the algorithm is long.  It is mostly based on
#          my earlier version at https://github.com/lehner/Grid/blob/legacy/lib/algorithms/iterative/ImplicitlyRestartedLanczos.h
#          which is based on code from Tom Blum, Taku Izubuchi, Chulwoo Jung and guided by the PhD thesis of Rudy Arthur.
#          I also adopted some of Peter Boyle's convergence test modifications that are in https://github.com/paboyle/Grid .
#
import gpt as g
import numpy as np
import math
from time import time

# Implicitly Restarted Lanczos
class irl:

    def __init__(self, params):
        self.params = params
        
    def __call__(self,mat,src):
        verbose="irl" in g.default.verbose

        # first approximate largest eigenvalue
        pit=g.algorithms.iterative.power_iteration({"eps" : 0.05,"maxiter": 10})
        lambda_max = pit(mat,src)[0]

        # parameters
        Nm=self.params["Nm"]
        Nk=self.params["Nk"]
        Nstop=self.params["Nstop"]

        # tensors
        dtype=np.float64
        lme=np.empty((Nm,),dtype)
        lme2=np.empty((Nm,),dtype)
        ev=np.empty((Nm,),dtype)
        ev2=np.empty((Nm,),dtype)
        ev2_copy=np.empty((Nm,),dtype)

        # fields
        f=g.lattice(src)
        v=g.lattice(src)
        evec=[ g.lattice(src) for i in range(Nm) ]

        # scalars
        k1=1
        k2=Nk
        Nconv=0
        beta_k=0.0

        # set initial vector
        evec[0] @= src / g.norm2(src)**0.5
        
        # initial Nk steps
        for k in range(Nk):
            self.step(mat,ev,lme,evec,f,Nm,k)

        # restarting loop
        for it in range(self.params["maxiter"]):
            if verbose:
                g.message("Restart iteration %d" % it)
            for k in range(Nk,Nm):
                self.step(mat,ev,lme,evec,f,Nm,k)
            f *= lme[Nm-1]

            # eigenvalues
            for k in range(Nm):
                ev2[k] = ev[k+k1-1]
                lme2[k] = lme[k+k1-1]

            # diagonalize
            t0=time()
            Qt=np.identity(Nm,dtype)
            self.diagonalize(ev2,lme2,Nm,Qt)
            t1=time()

            if verbose:
                g.message("Diagonalization took %g s" % (t1-t0))

            # sort
            ev2_copy=ev2.copy()
            ev2=list(reversed(sorted(ev2)))

            # implicitly shifted QR transformations
            Qt=np.identity(Nm,dtype)
            t0=time()
            for ip in range(k2,Nm):
                self.qr_decomp(ev,lme,Nm,Nm,Qt,ev2[ip],k1,Nm)
            t1=time()

            if verbose:
                g.message("QR took %g s" % (t1-t0))

            # rotate
            t0=time()
            g.rotate(evec,Qt,k1-1,k2+1,0,Nm,Nm)
            t1=time()

            if verbose:
                g.message("Basis rotation took %g s" % (t1-t0))

            # compression
            f *= Qt[k2-1,Nm-1]
            f += lme[k2-1] * evec[k2]
            beta_k = g.norm2(f)**0.5
            betar = 1.0/beta_k
            evec[k2] @= betar * f
            lme[k2-1] = beta_k

            if verbose:
                g.message("beta_k = ", beta_k)

            # convergence test
            if it >= self.params["Nminres"]:
                if verbose:
                    g.message("Rotation to test convergence")

                # diagonalize
                for k in range(Nm):
                    ev2[k] = ev[k]
                    lme2[k] = lme[k]
                Qt = np.identity(Nm,dtype)

                t0=time()
                self.diagonalize(ev2,lme2,Nk,Qt)
                t1=time()

                if verbose:
                    g.message("Diagonalization took %g s" % (t1-t0))

                B=g.copy(evec[0])
                
                allconv=True
                jj=1
                while jj<=Nstop:
                    j=Nstop-jj
                    g.rotateJ(B,evec,Qt,j,0,Nk,Nm)
                    B *= 1.0/g.norm2(B)**0.5
                    mat(B,v)
                    ev_test=g.innerProduct(B,v).real
                    eps2 = g.norm2(v - ev_test*B) / lambda_max**2.0
                    if verbose:
                        g.message("%-65s %-45s %-50s" % ("ev[ %d ] = %s" % (j,ev2_copy[j]),
                                                         "<B|M|B> = %s" % (ev_test),
                                                         "|M B - ev B|^2 / ev_max^2 = %s" % (eps2)))
                    if eps2 > self.params["resid"]:
                        allconv=False
                    if jj == Nstop:
                        break
                    jj=min([Nstop,2*jj])

                if allconv:
                    if verbose:
                        g.message("Converged in %d iterations" % it)

                    t0=time()
                    g.rotate(evec,Qt,0,Nstop,0,Nk,Nm)
                    t1=time()

                    if verbose:
                        g.message("Final basis rotation took %g s" % (t1-t0))

                    return (True,evec[0:Nstop],ev2_copy[0:Nstop])
                    
        if verbose:
            g.message("Did not converge")
        return (False,[],[])

    def diagonalize(self,lmd,lme,Nk,Qt):
        TriDiag = np.zeros((Nk,Nk),dtype=Qt.dtype)
        for i in range(Nk):
            TriDiag[i,i]=lmd[i]
        for i in range(Nk-1):
            TriDiag[i,i+1]=lme[i]
            TriDiag[i+1,i]=lme[i]
        w,v=np.linalg.eigh(TriDiag)
        for i in range(Nk):
            lmd[Nk-1-i] = w[i]
            for j in range(Nk):
                Qt[Nk-1-i,j]=v[j,i]

    def qr_decomp_fast(self,lmd,lme,Nk,Nm,Qt,Dsh,kmin,kmax):
        cgpt.qr_decomp(lmd,lme,Nk,Nm,Qt,Dsh,kmin,kmax)

    def qr_decomp(self,lmd,lme,Nk,Nm,Qt,Dsh,kmin,kmax):
        k = kmin-1    
        Fden = 1.0/math.hypot(lmd[k].real-Dsh.real,lme[k].real)
        c = ( lmd[k] -Dsh) *Fden
        s = -lme[k] *Fden
        tmpa1 = lmd[k]
        tmpa2 = lmd[k+1]
        tmpb  = lme[k]
        lmd[k]   = c*c*tmpa1 +s*s*tmpa2 -2.0*c*s*tmpb
        lmd[k+1] = s*s*tmpa1 +c*c*tmpa2 +2.0*c*s*tmpb
        lme[k]   = c*s*(tmpa1-tmpa2) +(c*c-s*s)*tmpb
        x        =-s*lme[k+1]
        lme[k+1] = c*lme[k+1]
        for i in range(Nk):
            Qtmp1 = Qt[k,i]
            Qtmp2 = Qt[k+1,i]
            Qt[k,i]  = c*Qtmp1 - s*Qtmp2
            Qt[k+1,i]= s*Qtmp1 + c*Qtmp2

        for k in range(kmin,kmax-1):
            Fden = 1.0/math.hypot(x.real,lme[k-1].real)
            c = lme[k-1]*Fden
            s = - x*Fden
            tmpa1 = lmd[k]
            tmpa2 = lmd[k+1]
            tmpb  = lme[k]
            lmd[k]   = c*c*tmpa1 +s*s*tmpa2 -2.0*c*s*tmpb
            lmd[k+1] = s*s*tmpa1 +c*c*tmpa2 +2.0*c*s*tmpb
            lme[k]   = c*s*(tmpa1-tmpa2) +(c*c-s*s)*tmpb
            lme[k-1] = c*lme[k-1] -s*x
            if k != kmax-2:
                x = -s*lme[k+1]
                lme[k+1] = c*lme[k+1]

            for i in range(Nk):
                Qtmp1 = Qt[k,i]
                Qtmp2 = Qt[k+1,i]
                Qt[k,i]     = c*Qtmp1 -s*Qtmp2
                Qt[k+1,i]   = s*Qtmp1 +c*Qtmp2


    def step(self,mat,lmd,lme,evec,w,Nm,k):
        assert(k<Nm)

        verbose="irl" in g.default.verbose

        alph=0.0
        beta=0.0

        evec_k=evec[k]

        t0=time()
        mat(evec_k,w)
        t1=time()

        if k > 0:
            w -= lme[k-1]*evec[k-1]

        zalph=g.innerProduct(evec_k,w)
        alph=zalph.real
        
        w -= alph * evec_k

        beta=g.norm2(w)**0.5
        w /= beta

        t2=time()
        if k>0:
            g.orthogonalize(w,evec[0:k])
        t3=time()

        if verbose:
            g.message("%-65s %-45s %-50s" % ("alpha[ %d ] = %s" % (k,zalph),
                                          "beta[ %d ] = %s" % (k,beta),
                                          " timing: %g s (matrix), %g s (ortho)" % (t1-t0,t3-t2)))
        
        lmd[k]=alph
        lme[k]=beta

        if k < Nm-1:
            evec[k+1] @= w
