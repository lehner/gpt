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
def exp(x,order = 12):
    o=gpt.lattice(x)
    o[:]=0
    nfac=1.0
    xn=gpt.copy(x)
    o[:]=numpy.identity(o.otype.shape[0],o.grid.precision.complex_dtype)
    o += xn
    for i in range(2,order + 1):
        nfac /= i
        xn @= xn * x
        o += xn * nfac
    return o
