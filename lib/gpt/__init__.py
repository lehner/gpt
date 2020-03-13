#
# GPT
#
# Authors: Christoph Lehner 2020
#
from gpt.grid import grid
from gpt.precision import single, double
from gpt.lattice import lattice, meminfo
from gpt.log import message
from gpt.transform import cshift, copy, norm2, innerProduct, axpy_norm
from gpt.expr import expr, expr_eval, adj, transpose, conj, trace, sum
from gpt.otype import *
from gpt.io import load
import gpt.default
import gpt.util
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
