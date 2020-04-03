#
#    GPT - Grid Python Toolkit
#    Copyright (C) 2020  Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)
#
#    This program is free software; you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation; either version 2 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License along
#    with this program; if not, write to the Free Software Foundation, Inc.,
#    51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
#
import gpt
import numpy

###
# Helper
def gpt_object(first, ot):
    if type(first) == gpt.grid:
        return gpt.lattice(first, ot)
    elif type(first) == list or type(first) == numpy.ndarray:
        return gpt.tensor(numpy.array(first, dtype=numpy.complex128), ot)

###
# Types below

class ot_complex:
    nfloats=2
    shape=()
    transposed=None
    spintrace=(None,None,None) # do nothing
    colortrace=(None,None,None)

def complex(grid):
    return gpt_object(grid, ot_complex)


class ot_mcolor:
    nfloats=2*3*3
    shape=(3,3)
    transposed=(1,0)
    spintrace=(None,None,None) # do nothing
    colortrace=(0,1,ot_complex)

def mcolor(grid):
    return gpt_object(grid, ot_mcolor)


class ot_vcolor:
    nfloats=2*3
    shape=(3,)
    transposed=None
    spintrace=None # not supported
    colortrace=None

def vcolor(grid):
    return gpt_object(grid, ot_vcolor)


class ot_mspincolor:
    nfloats=2*3*3*4*4
    shape=(4,4,3,3)
    transposed=(1,0,3,2)
    spintrace=(0,1,ot_mcolor)
    colortrace=None # not supported, due to current lack of ot_mspin

def mspincolor(grid):
    return gpt_object(grid, ot_mspincolor)


class ot_vspincolor:
    nfloats=2*3*4
    shape=(4,3)
    transposed=None
    spintrace=None
    colortrace=None

def vspincolor(grid):
    return gpt_object(grid, ot_vspincolor)


###
# String conversion for safe file input
def str_to_otype(s):
    return { 
        "ot_complex" : ot_complex,
        "ot_mcolor" : ot_mcolor,
        "ot_vcolor" : ot_vcolor,
        "ot_mspincolor" : ot_mspincolor,
        "ot_vspincolor" : ot_vspincolor,
        }[s]

###
# Multiplication table
mtab = {
    (ot_mcolor,ot_mcolor) : (ot_mcolor,(1,0)),
    (ot_mcolor,ot_vcolor) : (ot_vcolor,(1,0)),
    (ot_mspincolor,ot_mspincolor) : (ot_mspincolor,([1,3],[0,2])),
    (ot_mspincolor,ot_vspincolor) : (ot_vspincolor,([1,3],[0,1])),
}

###
# Outer product table
otab = {
    (ot_vcolor,ot_vcolor) : (ot_mcolor,[]),
    (ot_vspincolor,ot_vspincolor) : (ot_mspincolor,[(1,2)])
}

###
# Inner product table
itab = {
    (ot_vcolor,ot_vcolor) : (ot_complex,(0,0)),
    (ot_vspincolor,ot_vspincolor) : (ot_complex,([0,1],[0,1])),
}

