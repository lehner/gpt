#
#    GPT - Grid Python Toolkit
#    Copyright (C) 2026  Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)
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


def blocked_links(U, sublattice):
    # V_mu(c) = U_mu(b c) U_mu(b c + mu) ... U_mu(b c + (b_mu - 1) mu): the
    # links of the coarse grid of g.block.sublattice (products of group
    # elements keep the otype of U); the fine links are picked before they
    # are multiplied (no fine products or shifts), plain fields or reverse-AD
    # nodes
    nd = len(U)
    V = []
    for mu in range(nd):
        offset = [0] * nd
        v = None
        for k in range(sublattice.factor[mu]):
            offset[mu] = k
            r = sublattice.restrict(offset)(U[mu])
            v = r if v is None else g(v * r)
        V.append(v)
    return V
