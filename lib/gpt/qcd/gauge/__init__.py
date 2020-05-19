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
import gpt as g
import numpy as np

def plaquette(U):
    # U[mu](x)*U[nu](x+mu)*adj(U[mu](x+nu))*adj(U[nu](x))
    tr=0.0
    vol=float(U[0].grid.fsites)
    for mu in range(4):
        for nu in range(mu):
            tr += g.sum( g.trace(U[mu] * g.cshift( U[nu], mu, 1) * g.adj( g.cshift( U[mu], nu, 1 ) ) * g.adj( U[nu] )) )
    return 2.*tr.real/vol/4./3./3.

def unit(first):
    if type(first) == g.grid:
        U=[ g.mcolor(first) for i in range(4) ]
        unit(U)
        return U
    elif type(first) == list:
        for x in first:
            unit(x)
    elif type(first) == g.lattice:
        first[:]=g.mcolor(np.identity(3,dtype=first.grid.precision.complex_dtype))
    else:
        assert(0)

def random(first, rng, scale = 1.0):
    if type(first) == g.grid:
        U=[ g.mcolor(first) for i in range(4) ]
        random(U, rng, scale)
        return U
    elif type(first) == list:
        for x in first:
            random(x, rng, scale)
    elif type(first) == g.lattice:
        rng.lie(first, scale)
    else:
        assert(0)
