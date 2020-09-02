import gpt
from gpt.core.matrix import exp as mexp

from gpt.algorithms.integrators.molecular_dynamics import leap_frog


class update:
    def __init__(self, first, second):
        if not type(first) is list:
            self.fld = [first]
        else:
            self.fld = first

        if not type(second) is list:
            tmp = [second]
        else:
            tmp = second

        if hasattr(tmp[0], "force"):
            self.act = tmp
        else:
            self.mom = tmp

    def get_type(self):
        if self.fld[0].otype is gpt.otype.ot_matrix_su3_fundamental:
            return 0
        elif self.fld[0].otype is gpt.otype.ot_singlet:
            return 1

    def __call__(self, eps):
        t0 = gpt.time()
        for mu in range(len(self.fld)):
            if hasattr(self, "mom"):
                if self.get_type() == 0:
                    self.fld[mu] @= mexp(gpt.eval(eps * self.mom[mu])) * self.fld[mu]
                elif self.get_type() == 1:
                    self.fld[mu] += eps * self.mom[mu]
            else:
                for a in self.act:
                    a.pre_force()
                    frc = a.force(mu)
                    self.fld[mu] -= eps * frc
        return gpt.time() - t0

    def get_act(self):
        if hasattr(self, "act"):
            return self.act
        return []
