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


def field_type(field):
    fermion = field[0].islower()
    if fermion:
        if field[-3:] == "bar":
            return field[:-3], -1
        else:
            return field, 1
    return field, 0


def contract_factor(fac):
    if len(fac) == 0:
        return [(1, [])]
    elif len(fac) % 2 == 1:
        return [(0, [])]
    else:
        # remove first factor and contract it with all others
        field, coordinate = fac[0]
        field_n, field_t = field_type(field)
        parity = 1
        idx = 1
        res = []
        for p_field, p_coordinate in fac[1:]:
            p_field_n, p_field_t = field_type(p_field)
            if field_t == -p_field_t and field_n == p_field_n:
                fac_prime = [fac[i] for i in range(1, len(fac)) if i != idx]
                res_prime = contract_factor(fac_prime)
                for rf, re in res_prime:
                    if field_t == 0:
                        res.append((rf, re + [(field_n, coordinate, p_coordinate)]))
                    elif field_t == 1:
                        res.append((rf * parity, re + [(field_n, coordinate, p_coordinate)]))
                    elif field_t == -1:
                        res.append((-rf * parity, re + [(field_n, p_coordinate, coordinate)]))
                    else:
                        raise ValueError("Unknown field type")
            if p_field_t != 0:
                parity *= -1
            idx += 1
        return res


def contract(self):
    graph = []
    for c, p, fac in self.graph:
        facs = contract_factor(fac)
        for rf, re in facs:
            if abs(rf * c) > 1e-13:
                graph.append((rf * c, re))
    return g.qcd.feynman.diagrams(graph)
