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


def identity(U, mu=3):
    # U'(x) = V(x) U_mu(x) Vdag(x+mu)
    # V = [1, U[0], U[0] U[1], U[0] U[1] U[2], ...]
    U_n = g.separate(U[mu], mu)
    V_n = [g.identity(U_n[0])]
    N = len(U_n)
    for n in range(N - 1):
        V_n.append(g(V_n[n] * U_n[n]))
    return g.merge(V_n, mu)
