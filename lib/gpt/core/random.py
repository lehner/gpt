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
import gpt, cgpt, numpy

_seeded=False

def seed(s):
    global _seeded
    cgpt.random_seed(s)
    _seeded=True

def sample(t,p):
    if type(t) == list:
        for x in t:
            sample(x,p)
    elif t is None or type(t) == numpy.ndarray:
        x=cgpt.random_sample(t,p)
        return x
    elif type(t) == gpt.lattice:
        pos=gpt.coordinates(t.grid)
        t[pos]=sample(pos,{**p,**{"shape": list(t.otype.shape)} })
        return t
    elif type(t) == gpt.vlattice:
        for x in t.v:
            sample(x,p)
    elif type(t) == gpt.grid or type(t) == gpt.cartesian_view:
        return sample(gpt.coordinates(t),p)
    else:
        assert(0)

def normal(t = None,p = { "mu" : 0.0, "sigma" : 1.0 }):
    assert(_seeded)
    return sample(t,{**{ "distribution" : "normal" }, **p})

def cnormal(t = None,p = { "mu" : 0.0, "sigma" : 1.0 }):
    assert(_seeded)
    return sample(t,{**{ "distribution" : "cnormal" }, **p})

def uniform_real(t = None,p = { "min" : 0.0, "max" : 1.0 }):
    assert(_seeded)
    return sample(t,{**{ "distribution" : "uniform_real" }, **p})

def uniform_int(t = None,p = { "min" : 0, "max" : 1 }):
    assert(_seeded)
    return sample(t,{**{ "distribution" : "uniform_int" }, **p})

def zn(t = None,p = { "n" : 2 }):
    assert(_seeded)
    return sample(t,{**{ "distribution" : "zn" }, **p})
