#
# GPT
#
# Authors: Christoph Lehner 2020
#
from gpt.core import *
import gpt.default
import gpt.create
import gpt.algorithms
import gpt.qcd
import cgpt, sys

# initialize cgpt when gpt is loaded
cgpt.init(sys.argv)

# synonyms
eval=expr_eval

# global rank
def rank():
    return cgpt.global_rank()
