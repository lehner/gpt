#
#    GPT - Grid Python Toolkit
#    Copyright (C) 2025  Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)
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


def topology(U, Q_mean, Q_std, sin2_Pi_Q_poly_coefficients=[]):
    adU = [g.ad.reverse.node(g.copy(u)) for u in U]
    dQ = g.qcd.gauge.differentiable_topology(adU)
    dA = (dQ - Q_mean) * (dQ - Q_mean) * (1.0 / 2.0 / Q_std / Q_std)

    for n, c in enumerate(sin2_Pi_Q_poly_coefficients):
        if abs(c) < 1e-50:
            continue

        dSinPiQ = g.component.sin((np.pi * (n + 1)) * dQ)
        dSin2PiQ = dSinPiQ * dSinPiQ

        dA = dA - c * dSin2PiQ

    return dA.functional(*adU)


def topology_field(U, Q_std):
    adU = [g.ad.reverse.node(g.copy(u)) for u in U]
    adQf = g.ad.reverse.node(g.real(U[0].grid))
    
    dQ = g.qcd.gauge.differentiable_topology(adU, field=True)
    dA = g.norm2(g.astype(dQ, adQf.value.otype) - adQf) * (1.0 / 2.0 / Q_std / Q_std)

    return dA.functional(*adU, adQf)
