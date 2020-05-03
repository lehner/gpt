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
import cgpt, gpt, numpy

def cshift(first, second, third, fourth = None):

    if type(first) == gpt.lattice and type(second) == gpt.lattice and not fourth is None:
        t=first
        l=gpt.eval(second)
        d=third
        o=fourth
    else:
        l=gpt.eval(first)
        d=second
        o=third
        t=gpt.lattice(l)

    for i in t.otype.v_idx:
        cgpt.cshift(t.v_obj[i],l.v_obj[i],d,o)
    return t

def copy(first, second = None):

    if type(first) == gpt.lattice:
        if not second is None:
            t=first
            l=second
        else:
            l=first
            t=gpt.lattice(l)
        for i in t.otype.v_idx:
            cgpt.copy(t.v_obj[i],l.v_obj[i])
        return t

    else:
        assert(0)

def convert(first, second):
    if type(first) == gpt.lattice and type(second) == gpt.lattice:
        assert(len(first.otype.v_idx) == len(second.otype.v_idx))
        for i in first.otype.v_idx:
            cgpt.convert(first.v_obj[i],second.v_obj[i])
        return first
    elif second == gpt.single or second == gpt.double:
        if type(first) == list:
            src_grid=first[0].grid
        else:
            src_grid=first.grid
        dst_prec=second
        dst_grid=gpt.grid(src_grid.fdimensions,dst_prec,src_grid.cb)
        if type(first) == list:
            dst = [ convert(gpt.lattice(dst_grid, src.otype),src) for src in first ]
        else:
            src = first
            dst = convert(gpt.lattice(dst_grid, src.otype),src)
        return dst
    else:
        assert(0)

def innerProduct(a,b, rank_only = False):
    if type(a) == gpt.tensor and type(b) == gpt.tensor:
        return gpt.adj(a) * b
    a=gpt.eval(a)
    b=gpt.eval(b)
    assert(len(a.otype.v_idx) == len(b.otype.v_idx))
    r=sum([ cgpt.lattice_rankInnerProduct(a.v_obj[i],b.v_obj[i]) for i in a.otype.v_idx ])
    if rank_only:
        return r
    # do global sum only once not for each v_idx
    return a.grid.globalsum(rankInnerProduct(a,b))

def rankInnerProduct(a,b):
    return innerProduct(a,b, rank_only = True)

def norm2(l):
    return innerProduct(l,l).real

def innerProductNorm2(a,b):
    if type(a) == gpt.tensor and type(b) == gpt.tensor:
        return gpt.adj(a) * b, a.norm2()
    a=gpt.eval(a)
    b=gpt.eval(b)
    assert(len(a.otype.v_idx) == len(b.otype.v_idx))
    r=[ cgpt.lattice_innerProductNorm2(a.v_obj[i],b.v_obj[i]) for i in a.otype.v_idx ]
    return sum([ x[0] for x in r ]), sum([ x[1] for x in r ]) # todo, make local version of this too

def axpy_norm2(d, a, x, y):
    x=gpt.eval(x)
    y=gpt.eval(y)
    assert(len(y.otype.v_idx) == len(x.otype.v_idx))
    assert(len(d.otype.v_idx) == len(x.otype.v_idx))
    return sum([ cgpt.lattice_axpy_norm2(d.v_obj[i],a,x.v_obj[i],y.v_obj[i]) for i in x.otype.v_idx ])

def slice(x,dim):
    x=gpt.eval(x)
    r=sum([ numpy.array(cgpt.lattice_slice(o,dim)) for o in x.v_obj ])
    return [ gpt.util.value_to_tensor(v,x.otype) for v in r ]
