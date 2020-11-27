
# S = (phi, (M Mdag)^-1 phi) = (psi, psi)
#
# dS = -(phi, (M Mdag)^-1 (dM Mdag + M dMdag) (M Mdag)^-1 phi)
#    = -(psi, (M^-1 dM + dMdag Mdag^-1) psi)
#    = -2 Re[ (Mdag^-1 psi, dM psi) ] = -2 Re[ (chi, dM psi) ]
#
# chi = Mdag^-1 psi = (M Mdag)^-1 phi
# psi = M^-1 phi    = Mdag chi
#
# random initialization
# i) random eta from exp[ -(eta,eta) ]
# ii) phi = M eta, such that exp[ -(M^-1 phi, M^-1 phi) ]

# Mattia: design choice to not allocate pf field inside action
#         because SMD evolves it, so we can do both HMC and SMD

# Mattia: Smearing or Boundary conditions?

# Mattia: when we define operator, say wilson_clover we pass
#         U link. Then the HMC modifies the U links. We need
#         to assume that operator has updated itself? Or do 
#         we force it here? I mean update the doublegaugefield 
#         inside grid
#

import gpt

class doublet:
    def __init__(self,pf,M,Sinv,Finv,Finv2=None):
        # do we want some checks on the types, e.g. pf is spinorfield?
        self.pf = pf
        self.M = M
        self.Sinv = Sinv(M) * M.ImportPhysicalFermionSource
        if Finv2 is None:
            self.Finv = Finv(M * M.adj())
            self.Finv2 = None
        else:
            self.Finv = Finv(self.M)
            self.Finv2 = Finv2(self.M.adj())
        self.grid = M.F_grid
        self.Nc = M.U[0].otype.Nc
        self.frc = [gpt.lattice(U) for U in self.M.U] # --> bad for memory management?
        
    def __call__(self,rng=None):
        if not rng is None:
            eta = gpt.vspincolor(self.grid)
            rng.normal(eta) # exp(-0.5*eta^dag eta)
            self.pf @= self.M * eta
            return gpt.norm2(eta)
        else:
            psi = gpt.vspincolor(self.grid)
            gpt.eval(psi, self.Sinv * self.pf)
            return gpt.norm2(psi)
    
    def setup_force(self):
        self.M.update(self.M.U)
        self.chi = gpt.lattice(self.pf)
        psi = gpt.lattice(self.pf)
        if self.Finv2 is None:
            gpt.eval(self.chi, self.Finv * self.pf)
            psi @= self.M.adj() * self.chi
        else:
            gpt.eval(psi, self.Finv * self.pf)
            gpt.eval(self.chi, self.Finv2 * psi)
        
        #self.frc = [gpt.lattice(U) for U in self.M.U]
        tmp = [gpt.lattice(U) for U in self.M.U]
        
        # (psi, dMdag chi) ; NOTE False = DaggerNo in grid
        self.M.deriv(self.frc, psi, self.chi, True)
        
        # (chi, dM psi)
        self.M.deriv(tmp, self.chi, psi, False) 
        for mu in range(self.grid.nd):
            self.frc[mu] += tmp[mu]
        
        # in the future we can think to cache chi and use it to precondition the solver
        del self.chi
       
        
    def sun2alg(self, link):
        # U -> 0.5*(U - U^dag) - 0.5/N * tr(U-U^dag)
        link -= gpt.adj(link)
        tr = gpt.eval(gpt.trace(link)) / self.Nc
        tmp = gpt.lattice(link)
        tmp[:] = gpt.mcolor([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        link -= tmp * tr
        link *= 0.5
            
    def force(self, field):        
        for U in self.M.U:
            if U.v_obj == field.v_obj:
                mu = self.M.U.index(U)
                self.sun2alg(self.frc[mu])
                return self.frc[mu]

    #def clean_force(self,keep=False):
    #    self.frc.clear() # this is a list, will this work?
    #    del self.psi
    #    if not keep:
    #        del self.chi
        # need to do some cleaning of self.psi , self.chi, self.frc
        