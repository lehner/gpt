#
# GPT
#
# Authors: Christoph Lehner 2020
#
import gpt

class ot_complex:
    nfloats=2

def complex(grid):
    return gpt.lattice(grid, ot_complex)


class ot_mcolor:
    nfloats=2*3*3

def mcolor(grid):
    return gpt.lattice(grid, ot_mcolor)


class ot_vcolor:
    nfloats=2*3

def vcolor(grid):
    return gpt.lattice(grid, ot_vcolor)


class ot_mspincolor:
    nfloats=2*3*3*4*4

def mspincolor(grid):
    return gpt.lattice(grid, ot_mspincolor)


class ot_vspincolor:
    nfloats=2*3*4

def vspincolor(grid):
    return gpt.lattice(grid, ot_vspincolor)

# mspin, vspin
