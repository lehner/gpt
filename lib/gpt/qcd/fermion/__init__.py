#
# GPT
#
# Authors: Christoph Lehner 2020
#
import gpt.qcd.fermion.reference
import gpt.qcd.fermion.solver

from gpt.qcd.fermion.register import register
from gpt.qcd.fermion.operator import operator

# instantiate fermion operators
def wilson_clover(U, params):
    return operator("wilson_clover", U, params)
