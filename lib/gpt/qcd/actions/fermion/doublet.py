
# S = (phi, (M Mdag)^-1 phi) = (psi, psi)
#
# dS = -(phi, (M Mdag)^-1 (dM Mdag + M dMdag) (M Mdag)^-1 phi)
#    = -(psi, (M^-1 dM + dMdag Mdag^-1) psi)
#    = -2 Re[ (Mdag^-1 psi, dM psi) ] = -2 Re[ (chi, dM psi) ]
#
# chi = Mdag^-1 psi = (M Mdag)^-1 phi
# psi = Mdag chi    = M^-1 phi
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
            #self.Finv = Finv(M * M.adj)
            self.Finv2 = None
        else:
            self.Finv = Finv(self.M)
            self.Finv2 = Finv2(self.M.adj)
        self.grid = M.F_grid
        
    def refresh(self, rng):
        #
        eta = gpt.vspincolor(self.grid)
        rng.normal(eta)
        self.pf @= self.M * eta
        
    def __call__(self):
        psi = gpt.vspincolor(self.grid)
        gpt.eval(psi, self.Sinv * self.pf)
        return gpt.norm2(psi)
    
    def setup_force(self):
        #self.M.update(op.U) ---> ?
        if self.Finv2 is None:
            gpt.eval(self.chi, self.Finv * self.pf)
            self.psi @= self.M.adj * self.chi
        else:
            gpt.eval(self.psi, self.Finv * self.pf)
            gpt.eval(self.chi, self.Finv2 * self.psi)
            
        self.frc = [gpt.lattice(U) for U in self.M.U]
        tmp = [gpt.lattice(U) for U in self.M.U]
        
        self.M.deriv(self.frc, self.psi, self.chi, False)
        self.M.deriv(tmp, self.chi, self.psi, True)
        self.frc += tmp
        
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
                return self.sun2alg(self.frc[mu])

    def clean_force(self,keep=False):
        self.frc.clear() # this is a list, will this work?
        del self.psi
        if not keep:
            del self.chi
        # need to do some cleaning of self.psi , self.chi, self.frc
        