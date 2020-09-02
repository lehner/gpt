import gpt
import numpy


class conjugate_momenta:
    def __init__(self, field):
        if type(field) is list:
            self.mom = [gpt.lattice(f) for f in field]
        else:
            self.mom = [gpt.lattice(field)]

    def action(self):
        # -1/2 pi^a pi^a, with pi^a from the algebra;
        # pi = sum_a Ta pi^a,  - tr(pi*pi) = - pi^a pi^b tr(Ta Tb) = - pi^a pi^b (-1/2) delta_{ab} = 1/2 pi^a pi^a
        # since convention is tr(Ta Tb) = -1/2 delta_{ab}, we do not include 1/2 in action
        momsq = 0
        for m in self.mom:
            momsq += gpt.sum(gpt.trace(m * m))
        if self.mom[0].otype is gpt.otype.ot_singlet:
            return 0.5 * momsq.real
        return -momsq.real

    def refresh(self, rng):
        ca = gpt.complex(self.mom[0].grid)
        ta = gpt.lattice(self.mom[0])

        for m in self.mom:
            m[:] = 0
            rng.normal(ca, {"mu": 0.0, "sigma": 1.0})

            if hasattr(m.otype, "generators"):
                for g in m.otype.generators(m.grid.precision.complex_dtype):
                    ta[:] = g
                    m += 1j * ca * ta
            else:
                m += ca

    def reverse(self):
        for m in self.mom:
            m *= -1.0


class hmc:
    def __init__(self, fld, mom, pf, mdint, rng):
        self.mdint = mdint
        self.rng = rng
        self.mom = mom
        if type(fld) is not list:
            self.fld = [fld]
        else:
            self.fld = fld
        self.fld_copy = [gpt.copy(f) for f in self.fld]
        self.pf = pf  # pseudo-fermion
        self.act = mdint.get_act()

    def __call__(self, tau):
        t0 = gpt.time()
        dts = -gpt.time()
        for mu in range(len(self.fld)):
            self.fld_copy[mu] @= self.fld[mu]
        self.mom.refresh(self.rng)
        if self.pf is not None:
            self.pf.refresh(self.rng)

        dts += gpt.time()

        dta = dti = 0.0
        dta -= gpt.time()
        a0 = [self.mom.action()]
        a0 += [a() for a in self.act]
        dta += gpt.time()

        dti -= gpt.time()
        self.mdint(tau)
        dti += gpt.time()

        dta -= gpt.time()
        a1 = [self.mom.action()]
        a1 += [a() for a in self.act]
        dta += gpt.time()

        # accept/reject
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

        gpt.message(
            f"HMC Trajectory generated in {gpt.time()-t0:g} secs; accept = {accept:d}; dH = {dH:g}"
        )
        gpt.message(
            f"HMC Timings = setup {dts:g} secs, actions {dta:g} secs, integrators {dti:g} secs"
        )

        return [accept, dH, numpy.exp(-dH)]

    def reversibility_test(self, tau):
        t0 = gpt.time()
        dts = -gpt.time()
        for mu in range(len(self.fld)):
            self.fld_copy[mu] @= self.fld[mu]
        self.mom.refresh(self.rng)
        if self.pf is not None:
            self.pf.refresh(self.rng)
        dts += gpt.time()

        dta = dti = 0.0
        dta -= gpt.time()
        a0 = [self.mom.action()]
        a0 += [a() for a in self.act]
        dta += gpt.time()

        dti -= gpt.time()
        self.mdint(tau)
        dti += gpt.time()

        dts -= gpt.time()
        self.mom.reverse()
        dts += gpt.time()

        dti -= gpt.time()
        self.mdint(tau)
        dti += gpt.time()

        dta -= gpt.time()
        a1 = [self.mom.action()]
        a1 += [a() for a in self.act]
        dta += gpt.time()

        # accept/reject
        dH = sum(a1) - sum(a0)
        gpt.barrier()

        gpt.message(f"HMC Reversibility test in {gpt.time()-t0:g} secs; dH = {dH:g}")
        return dH
