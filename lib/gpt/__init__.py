#
# GPT
#
# Authors: Christoph Lehner 2020
#
from gpt.grid import grid
from gpt.precision import single
from gpt.lattice import lattice, meminfo
from gpt.log import message
from gpt.transform import cshift, copy, mul, norm2, innerProduct, adj, axpy_norm
from gpt.expr_linear_combination import expr_linear_combination, eval
import gpt.otype, gpt.default, gpt.util, cgpt, sys, types

# initialize cgpt when gpt is loaded
cgpt.init(sys.argv)

# short-hand lattice definitions
def complex(grid):
    return lattice(grid, otype.complex)

# global rank
def rank():
    return cgpt.global_rank()
