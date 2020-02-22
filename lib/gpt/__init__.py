#
# GPT
#
# Authors: Christoph Lehner 2020
#
from gpt.grid import grid
from gpt.precision import single
from gpt.lattice import lattice
import gpt.otype

# short-hand lattice definitions
def complex(grid):
    return lattice(grid, otype.complex)
