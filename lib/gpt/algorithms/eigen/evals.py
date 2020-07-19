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


class EvalsNotConverged(Exception):
    pass


@g.params_convention(check_eps2=None, skip=1, real=False)
def evals(matrix, evec, params):
    check_eps2 = params["check_eps2"]
    skip = params["skip"]
    assert len(evec) > 0
    tmp = g.lattice(evec[0])
    ev = []
    for i in range(0, len(evec), skip):
        v = evec[i]
        matrix(tmp, v)
        # M |v> = l |v> -> <v|M|v> / <v|v>
        l = g.innerProduct(v, tmp) / g.norm2(v)
        if params["real"]:
            l = l.real
        ev.append(l)
        if check_eps2 is not None:
            eps2 = g.norm2(tmp - l * v)
            g.message(f"eval[ {i} ] = {l}, eps^2 = {eps2}")
            if eps2 > check_eps2:
                raise EvalsNotConverged()

    return ev
