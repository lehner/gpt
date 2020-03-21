#
# GPT
#
# Authors: Christoph Lehner 2020
#
import gpt.qcd.fermion.reference
import gpt.qcd.fermion.solver
import gpt.qcd.fermion.preconditioner

from gpt.qcd.fermion.register import register
from gpt.qcd.fermion.operator import operator

import copy

# instantiate fermion operators
def wilson_clover(U, params):
    params = copy.deepcopy(params) # save current parameters
    if "kappa" in params:
        assert(not "mass" in params)
        params["mass"] = (1./params["kappa"]/2. - 4.)
    return operator("wilson_clover", U, params)
