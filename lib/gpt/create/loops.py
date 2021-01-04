#
#    GPT - Grid Python Toolkit
#    Copyright (C) 2020  Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)
#                  2020  Lorenzo Barca    (lorenzo1.barca@ur.de)
#
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
#
import gpt as g

# gpt/qcd/gauge/loops in gpt/create or gpt/meas ?

def polyakov_loop(U, mu):
    # tr[ prod_j U_{\mu}(m, j) ]
    vol = float(U[0].grid.fsites)
    Nc = U[0].otype.Nc
    tmp_polyakov_loop = g.copy(U[mu])
    for n in range(1, U[0].grid.fdimensions[mu]):
        tmp = g.cshift(tmp_polyakov_loop, mu, 1)
        tmp_polyakov_loop = g.eval(U[mu] * tmp)

    return g.sum(g.trace(tmp_polyakov_loop)) / Nc / vol

