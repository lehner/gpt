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

class vlattice:
    def __init__(self, grid, n, fundamental):
        self.n = n
        self.grid = grid
        self.decomposition = decompose(n, fundamental.keys())
        self.n0,self.n1 = get_range(self.decomposition)
        self.v = [ fundamental[x](grid) for x in self.decomposition ]
        print(self.n0,self.n1)

    def __repr__(self):
        return "vlattice(%s,%s)" % ([ l.otype.__name__ for l in self.v ],self.grid.precision.__name__)
        
    def __str__(self):
        s=""
        for i,x in enumerate(self.v):
            s+="-------- %d to %d --------\n" % (self.n0[i],self.n1[i])
            s+=str(x)
        return s
