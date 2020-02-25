#
# GPT
#
# Authors: Christoph Lehner 2020
#
from gpt.grid import grid
from gpt.precision import single, double
from gpt.lattice import lattice, meminfo
from gpt.log import message
from gpt.transform import cshift, copy, mul, norm2, innerProduct, adj, axpy_norm
from gpt.expr_linear_combination import expr_linear_combination, eval
import gpt.otype
import gpt.default
import gpt.util
import gpt.algorithms
import cgpt, sys

# initialize cgpt when gpt is loaded
cgpt.init(sys.argv)

# short-hand lattice definitions
def complex(grid):
    return lattice(grid, otype.complex)

# global rank
def rank():
    return cgpt.global_rank()
