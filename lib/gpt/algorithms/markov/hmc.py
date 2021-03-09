#
#    GPT - Grid Python Toolkit
#    Copyright (C) 2020  Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)
#                  2020  Mattia Bruno
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

import gpt
import numpy


class conjugate_momenta:
    def __init__(self, field):
        if type(field) is list:
            self.mom = [gpt.lattice(f.grid,f.otype.cartesian()) for f in gpt.core.util.to_list(field)]
            self.fld = gpt.core.util.to_list(field)

        self.N = len(self.mom)

    def action(self):
        # -1/2 pi^a pi^a, with pi^a from the algebra;
        # pi = sum_a Ta pi^a, [convention is tr(Ta Tb) = -1/2 delta_{ab}]
        momsq = 0
        for m in self.mom:
            pia = m.otype.coordinates(m)
            for p in pia:
                momsq += gpt.sum(p*p);
#         if self.mom[0].otype is gpt.otype.ot_singlet:
#             return 0.5 * momsq.real
        return -0.5 * momsq.real

    def refresh(self, rng):
#         ca = gpt.complex(self.mom[0].grid)
#         ta = gpt.lattice(self.mom[0])

        for m in self.mom:
            rng.element(m, scale=1.0, normal=True)
#             m[:] = 0
#             if hasattr(m.otype, "generators"):
#                 for g in m.otype.generators(m.grid.precision.complex_dtype):
#                     ta[:] = g
#                     rng.normal(ca, {"mu": 0.0, "sigma": 1.0})
#                     ca[:].imag = 0.0
#                     m += 1j * ca * ta
#             else:
#                 rng.normal(ca, {"mu": 0.0, "sigma": 1.0})
#                 m += ca

    def reverse(self):
        for m in self.mom:
            m *= -1.0


class hmc:
    def __init__(self, fld, mom, pf, mdint, rng):
        self.mdint = mdint
        self.rng = rng
        self.mom = mom
        self.fld = gpt.core.util.to_list(fld)
        self.fld_copy = [gpt.copy(f) for f in self.fld]
        self.pf = pf  # pseudo-fermion
        self.act = mdint.get_act()

        
    def __call__(self, tau):
        verbose = gpt.default.is_verbose("hmc")
        time = gpt.timer("HMC")

        time("setup")
        for mu in range(len(self.fld)):
            self.fld_copy[mu] @= self.fld[mu]
        self.mom.refresh(self.rng)
        if self.pf is not None:
            self.pf.refresh(self.rng)

        time("actions")
        a0 = [self.mom.action()]
        a0 += [a() for a in self.act]

        time("integrators")
        self.mdint(tau)

        time("actions")
        a1 = [self.mom.action()]
        a1 += [a() for a in self.act]

        # accept/reject
        gpt.message(a0,a1)
        dH = sum(a1) - sum(a0)

        # decision taken on master node, but for completeness all nodes throw one random number
        rr = self.rng.uniform_real(None, {"min": 0, "max": 1})
        accept = 0
        if gpt.rank() == 0:
            accept = 1
            if dH > 0.0:
                if numpy.exp(-dH) < rr:
                    accept = 0
        gpt.barrier()
        accept = self.fld[0].grid.globalsum(accept)

        if accept == 0:
            for mu in range(len(self.fld)):
                self.fld[mu] @= self.fld_copy[mu]

        if verbose:
            time()
            gpt.message(time)
            gpt.message(f"HMC Trajectory: accept = {accept:d}; dH = {dH:g}")

        return [accept, dH, numpy.exp(-dH)]

    
    def reversibility_test(self, tau):
        for mu in range(len(self.fld)):
            self.fld_copy[mu] @= self.fld[mu]
        self.mom.refresh(self.rng)
        if self.pf is not None:
            self.pf.refresh(self.rng)

        a0 = [self.mom.action()]
        a0 += [a() for a in self.act]

        self.mdint(tau)
        self.mom.reverse()
        self.mdint(tau)

        a1 = [self.mom.action()]
        a1 += [a() for a in self.act]

        # accept/reject
        dH = sum(a1) - sum(a0)
        gpt.barrier()

        gpt.message(f"HMC Reversibility test dH = {dH:g}")
        return dH
