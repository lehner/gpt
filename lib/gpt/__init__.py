#
# GPT
#
# Authors: Christoph Lehner 2020
#
from gpt.grid import grid
from gpt.precision import single
from gpt.lattice import lattice
from gpt.log import message
import gpt.otype, cgpt, sys

# initialize cgpt when gpt is loaded
cgpt.init(sys.argv)

# short-hand lattice definitions
def complex(grid):
    return lattice(grid, otype.complex)

# global rank
def rank():
    return cgpt.global_rank()
