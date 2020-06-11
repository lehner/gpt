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
def decompose(n, ns):
    r=[]
    for x in reversed(sorted(ns)):
        y=n // x
        r = r + [ x ]*y
        n = n % x
    if n != 0:
        raise Exception("Cannot decompose %d in available fundamentals %s" % (n,ns))
    return r

def get_range(ns):
    n=0
    n0=[]
    n1=[]
    for x in ns:
        n0.append(n)
        n+=x
        n1.append(n)
    return n0,n1

def gpt_object(first, ot):
    if type(first) == gpt.grid:
        return gpt.lattice(first, ot)
    elif type(first) == list or type(first) == numpy.ndarray:
        return gpt.tensor(numpy.array(first, dtype=numpy.complex128), ot)
    else:
        assert(0)

###
# Types below
class ot_base:
    v_otype=[ None ]
    v_n0=[ 0 ]
    v_n1=[ 1 ]
    v_idx=[ 0 ]
    transposed=None
    spintrace=None # not supported
    colortrace=None

class ot_complex(ot_base):
    nfloats=2
    shape=(1,)
    spintrace=(None,None,None) # do nothing
    colortrace=(None,None,None)
    v_otype=[ "ot_complex" ]

def complex(grid):
    return gpt_object(grid, ot_complex)


class ot_mcolor(ot_base):
    nfloats=2*3*3
    shape=(3,3)
    transposed=(1,0)
    spintrace=(None,None,None) # do nothing
    colortrace=(0,1,ot_complex)
    generators=lambda dt: [
        numpy.array([[ 0, 1, 0 ], [ 1, 0, 0 ], [ 0, 0, 0 ]], dtype=dt) / 2.0,
        numpy.array([[ 0, -1j, 0 ], [ 1j, 0, 0 ], [ 0, 0, 0 ]], dtype=dt) / 2.0,
        numpy.array([[ 1, 0, 0 ], [ 0, -1, 0 ], [ 0, 0, 0 ]], dtype=dt) / 2.0,
        numpy.array([[ 0, 0, 1 ], [ 0, 0, 0 ], [ 1, 0, 0 ]], dtype=dt) / 2.0,
        numpy.array([[ 0, 0, -1j ], [ 0, 0, 0 ], [ 1j, 0, 0 ]], dtype=dt) / 2.0,
        numpy.array([[ 0, 0, 0 ], [ 0, 0, 1 ], [ 0, 1, 0 ]], dtype=dt) / 2.0,
        numpy.array([[ 0, 0, 0 ], [ 0, 0, -1j ], [ 0, 1j, 0 ]], dtype=dt) / 2.0,
        numpy.array([[ 1, 0, 0 ], [ 0, 1, 0 ], [ 0, 0, -2 ]], dtype=dt) / ( 3.0 ) ** 0.5 / 2.0,
    ]
    v_otype=[ "ot_mcolor" ]

def mcolor(grid):
    return gpt_object(grid, ot_mcolor)


class ot_vcolor(ot_base):
    nfloats=2*3
    shape=(3,)
    v_otype=[ "ot_vcolor" ]

def vcolor(grid):
    return gpt_object(grid, ot_vcolor)


class ot_mspincolor(ot_base):
    nfloats=2*3*3*4*4
    shape=(4,4,3,3)
    transposed=(1,0,3,2)
    spintrace=(0,1,ot_mcolor)
    colortrace=None # not supported, due to current lack of ot_mspin
    v_otype=[ "ot_mspincolor" ]

def mspincolor(grid):
    return gpt_object(grid, ot_mspincolor)

class ot_vspincolor(ot_base):
    nfloats=2*3*4
    shape=(4,3)
    v_otype=[ "ot_vspincolor" ]

    def distribute(mat,dst,src,zero_lhs):
        if src.otype.__name__ == "ot_mspincolor":
            grid=src.grid
            dst_sc,src_sc=gpt.vspincolor(grid),gpt.vspincolor(grid)
            for s in range(4):
                for c in range(3):
                    gpt.qcd.prop_to_ferm(src_sc,src,s,c)
                    if zero_lhs:
                        dst_sc[:]=0
                    mat(dst_sc,src_sc)
                    gpt.qcd.ferm_to_prop(dst,dst_sc,s,c)
        else:
            assert(0)

def vspincolor(grid):
    return gpt_object(grid, ot_vspincolor)


###
# Basic vectors for coarse grid
class ot_vcomplex10(ot_base):
    nfloats=2*10
    shape=(10,)
    v_otype=[ "ot_vcomplex10" ]

class ot_vcomplex20(ot_base):
    nfloats=2*20
    shape=(20,)
    v_otype=[ "ot_vcomplex20" ]

class ot_vcomplex40(ot_base):
    nfloats=2*40
    shape=(40,)
    v_otype=[ "ot_vcomplex40" ]

class ot_vcomplex80(ot_base):
    nfloats=2*80
    shape=(80,)
    v_otype=[ "ot_vcomplex80" ]

class ot_vcomplex:
    fundamental={
        10 : ot_vcomplex10,
        20 : ot_vcomplex20,
        40 : ot_vcomplex40,
        80 : ot_vcomplex80,
    }
    def __init__(self, n):
        self.__name__="ot_vcomplex(%d)" % n
        self.nfloats=2*n
        self.shape=(n,)
        self.transposed=None
        self.spintrace=None
        self.colortrace=None
        decomposition=decompose(n, ot_vcomplex.fundamental.keys())
        self.v_n0,self.v_n1 = get_range(decomposition)
        self.v_idx=range(len(self.v_n0))
        self.v_otype = [ ot_vcomplex.fundamental[x] for x in decomposition ]

def vcomplex(grid, n):
    return gpt_object(grid, ot_vcomplex(n))

###
# String conversion for safe file input
def str_to_otype(s):
    base_types={ 
        "ot_complex" : ot_complex,
        "ot_mcolor" : ot_mcolor,
        "ot_vcolor" : ot_vcolor,
        "ot_mspincolor" : ot_mspincolor,
        "ot_vspincolor" : ot_vspincolor,
        "ot_vcomplex10" : ot_vcomplex10,
        "ot_vcomplex20" : ot_vcomplex20,
        "ot_vcomplex40" : ot_vcomplex40,
        "ot_vcomplex80" : ot_vcomplex80,
    }
    if s in base_types:
        return base_types[s]
    a=s.split("(")
    assert(len(a)==2)
    assert(a[1][-1]==")")
    base_vtypes={
        "ot_vcomplex" : ot_vcomplex
    }
    return base_vtypes[a[0]](int(a[1][:-1]))

###
# Construct otype from v_otype
def from_v_otype(v_otype):
    # split up v_otype in base and n
    base=list(set([ b.rstrip("0123456789") for b in v_otype ]))
    assert(len(base) == 1)
    base=base[0]
    decomposition=[ int(b[len(base):]) for b in v_otype ]
    n=sum(decomposition)
    return eval("gpt.otype.%s(%d)"  % (base,n))

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

