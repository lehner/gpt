#
# GPT
#
# Authors: Christoph Lehner 2020
#
import gpt

###
# Complex
class ot_complex:
    nfloats=2

ot_complex.SPINTRACE_OTYPE=ot_complex
ot_complex.COLORTRACE_OTYPE=ot_complex

###
# MColor
class ot_mcolor:
    nfloats=2*3*3

ot_mcolor.SPINTRACE_OTYPE=ot_mcolor
ot_mcolor.COLORTRACE_OTYPE=ot_complex

###
# VColor
class ot_vcolor:
    nfloats=2*3

ot_vcolor.SPINTRACE_OTYPE=None
ot_vcolor.COLORTRACE_OTYPE=None

###
# Short-hand lattice definitions
def complex(grid):
    return gpt.lattice(grid, ot_complex)

def mcolor(grid):
    return gpt.lattice(grid, ot_mcolor)

def vcolor(grid):
    return gpt.lattice(grid, ot_vcolor)

# mspin, vspin, mspincolor, vspincolor
