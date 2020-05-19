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
import gpt, numpy

# matrix exponential
def exp(i):
    x=i
    if i.grid.precision != gpt.double:
        x=gpt.convert(x, gpt.double)
    n=gpt.norm2(x)**0.5 / x.grid.fsites
    order=19
    maxn=0.05
    ns=0
    if n > maxn:
        ns=int(numpy.log2(n/maxn))
        x /= 2**ns
    o=gpt.lattice(x)
    o[:]=0
    nfac=1.0
    xn=gpt.copy(x)
    o[:]=numpy.identity(o.otype.shape[0],o.grid.precision.complex_dtype)
    o += xn
    for j in range(2,order + 1):
        nfac /= j
        xn @= xn * x
        o += xn * nfac
    for j in range(ns):
        o @= o*o
    if i.grid.precision != gpt.double:
        r=gpt.lattice(i)
        gpt.convert(r,o)
        o=r
    return o
