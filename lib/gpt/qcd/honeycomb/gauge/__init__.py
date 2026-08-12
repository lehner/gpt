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
import numpy as np


def cshift(fld, offset):
    for mu in range(4):
        if float(offset[mu]) != 0.0:
            assert abs(offset[mu] - int(offset[mu])) < 1e-10
            fld = g.cshift(fld, mu, int(offset[mu]))
    return fld


def plaquette(self, U):
    geo = self.geo
    assert len(U) == geo.number_of_link_fields
    Ptot = None
    for inst in geo.plaquette_instructions:
        P = None
        for subset, dag, offset, index in inst:
            fac = U[subset * 12 + index]
            if dag:
                fac = g(g.adj(fac))
            fac = cshift(fac, offset)
            if P is None:
                P = fac
            else:
                P = g(P * fac)
        if Ptot is None:
            Ptot = P
        else:
            Ptot += P
    return g(g.sum(g.trace(Ptot))).real / fac.grid.gsites / 3 / len(geo.plaquette_instructions)


def transformed(self, U, V):
    geo = self.geo
    assert len(V) == 2
    assert len(U) == geo.number_of_link_fields

    return [
        g(
            V[subset]
            * U[subset * 12 + idx]
            * g.adj(
                cshift(
                    V[(subset + geo.cross_grids[idx]) % 2],
                    geo.sub_offset[subset]
                    + np.array(geo.positive[idx])
                    - geo.sub_offset[(subset + geo.cross_grids[idx]) % 2],
                )
            )
        )
        for subset in [0, 1]
        for idx in range(len(geo.positive))
    ]
