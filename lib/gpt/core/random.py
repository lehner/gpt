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
import gpt, cgpt, numpy, sys

class random:

    def __init__(self, first, second = None):

        if type(first) == dict and second is None:
            s=first["seed"]
            engine=first["engine"]
        else:
            s=first
            engine=second
            if engine is None:
                engine="vectorized_ranlux24_389_64"

        self.verbose = gpt.default.is_verbose("random")
        t0=gpt.time()
        self.obj = cgpt.create_random(engine,s)
        t1=gpt.time()

        if self.verbose:
            gpt.message("Initializing gpt.random(%s,%s) took %g s" % (s,engine,t1-t0))

    def __del__(self):
        cgpt.delete_random(self.obj)

    def sample(self,t,p):
        if type(t) == list:
            for x in t:
                self.sample(x,p)
        elif t is None:
            return cgpt.random_sample(self.obj,t,p)
        elif type(t) == gpt.lattice:
            if "pos" in p:
                pos=p["pos"]
            else:
                pos=gpt.coordinates(t)
            t0=gpt.time()
            mv=cgpt.random_sample(self.obj,pos,{**p,**{"shape": list(t.otype.shape), "grid":t.grid.obj, "precision" : t.grid.precision} })
            t1=gpt.time()
            t[pos]=mv
            if self.verbose:
                szGB=mv.size * mv.itemsize / 1024.**3.
                gpt.message("Generated %g GB of random data at %g GB/s" % (szGB,szGB/(t1-t0)))
            return t
        else:
            assert(0)

    def normal(self,t = None,p = { "mu" : 0.0, "sigma" : 1.0 }):
        return self.sample(t,{**{ "distribution" : "normal" }, **p})

    def cnormal(self,t = None,p = { "mu" : 0.0, "sigma" : 1.0 }):
        return self.sample(t,{**{ "distribution" : "cnormal" }, **p})

    def uniform_real(self,t = None,p = { "min" : 0.0, "max" : 1.0 }):
        return self.sample(t,{**{ "distribution" : "uniform_real" }, **p})

    def uniform_int(self,t = None,p = { "min" : 0, "max" : 1 }):
        return self.sample(t,{**{ "distribution" : "uniform_int" }, **p})

    def zn(self,t = None,p = { "n" : 2 }):
        return self.sample(t,{**{ "distribution" : "zn" }, **p})

    def lie(self, out, scale = 1.0):
        grid=out.grid
        ca=gpt.complex(grid)
        lie=gpt.lattice(out)
        ta=gpt.mcolor(grid)

        lie[:]=0
        for g in out.otype.generators(grid.precision.complex_dtype):
            self.uniform_real(ca,{"min" : -0.5,"max" : 0.5})
            ta[:]=g
            lie += scale * 1j * ca * ta
        out @= gpt.matrix.exp(lie)

# sha256
def sha256(mv):
    if type(mv) == memoryview:
        a=cgpt.util_sha256(mv)
        r=a[0]
        for i in range(7):
            r=r*(2**32) + a[1+i]
        return r
    else:
        return sha256(memoryview(mv))
