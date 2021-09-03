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

import math

def line_search_quadratic(s, x, dx, dv0, df, step):
    x = g.util.to_list(x)
    xp = g.copy(x)
    # ansatz: f(x) = a + b*(x-c)^2, then solve for c from dv1 and dv0
    # assume b > 0
    sv0 = g.group.inner_product(s, dv0)
    assert not math.isnan(sv0)
    sign = 1
    if sv0 == 0.0:
        return 0.0
    elif sv0 < 0:
        sign = -1
    c = 0.0
    sv_list = [ sv0, ]
    while True:
        dxp = []
        for dx_mu, s_mu in g.util.to_list(dx, s):
            mu = x.index(dx_mu)
            xp[mu] @= g(g.group.compose(sign * step * s_mu, xp[mu]))
            xp_mu = g.copy(xp[mu])
            g.project(xp[mu], "defect")
            project_diff2 = g.norm2(xp[mu] - xp_mu)
            if not (project_diff2 < 1e-8):
                g.message(f"line_search_quadratic: rank={g.rank()} project_diff={math.sqrt(project_diff2)} {sv_list}")
                if c == 0.0:
                    return float("nan")
                else:
                    return sign * c
            dxp.append(xp[mu])

        dv1 = df(xp, dxp)
        assert isinstance(dv1, list)

        sv1 = g.group.inner_product(s, dv1)
        sv_list.append(sv1)
        if len(sv_list) > 10:
            g.message(f"line_search_quadratic: rank={g.rank()} {sv_list}")
        if math.isnan(sv1):
            g.message(f"line_search_quadratic: rank={g.rank()} {sv_list}")
            return float("nan")
        if sv0 > 0 and sv1 <= 0 or sv0 < 0 and sv1 >= 0:
            c += sv0 / (sv0 - sv1)
            return sign * c
        elif sv0 == 0.0:
            return sign * c
        else:
            c += 1
            sv0 = sv1

def line_search_none(s, x, dx, dv0, df, step):
    return 1.0
