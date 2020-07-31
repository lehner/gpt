#
#    GPT - Grid Python Toolkit
#    Copyright (C) 2020  Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)
#                  2020 Tilo Wettig
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


def fundamental_to_adjoint(U_a, U_f):
    """
    Convert fundamental to adjoint representation.  For now only SU(2) is supported.

    Input: fundamental gauge field

    Output: adjoint gauge field
    """
    grid = U_f.grid
    T = U_f.otype.generators(grid.precision.complex_dtype)
    V = {}
    for a in range(len(T)):
        for b in range(len(T)):
            V[a, b] = g.eval(2.0 * g.trace(T[a] * U_f * T[b] * g.adj(U_f)))
    g.merge_color(U_a, V)
