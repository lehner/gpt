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
from feynman import Diagram


def draw(ax, cmd, ops, flavors):
    cmd = [ln.split("_") for ln in cmd.split("/")]
    v_used = set()
    for f, a, b in cmd:
        if a not in ops:
            raise ValueError(f"{a} not defined")
        if b not in ops:
            raise ValueError(f"{b} not defined")
        v_used.add(a)
        v_used.add(b)
        if f not in flavors:
            raise ValueError(f"{f} not defined")

    diagram = Diagram(ax)

    v = {}
    for op in v_used:
        a, b, t, dx, dy = ops[op]
        v[op] = diagram.vertex(xy=(a, b))
        v[op].scale(0.5)
        v[op].text(t, dx, dy)

    pairs = {}
    pairs_partial = {}

    for f, a, b in cmd:
        pair = (a, b) if a < b else (b, a)
        if pair not in pairs:
            pairs[pair] = 1
            pairs_partial[pair] = 0
        else:
            pairs[pair] += 1

    for f, a, b in cmd:
        pair = (a, b) if a < b else (b, a)
        if pairs[pair] % 2 == 1:
            pp = pairs_partial[pair]
            if pp == 0:
                el = 0
            else:
                el = (-1) ** pp * (0.15 * ((pp - 1) // 2 + 1)) * (-1 if a < b else 1)
            pairs_partial[pair] = pp + 1
        else:
            pp = pairs_partial[pair]
            el = (-1) ** pp * (0.15 * (pp // 2 + 1)) * (-1 if a < b else 1)
            pairs_partial[pair] = pp + 1

        if el != 0:
            diagram.line(
                v[b], v[a], arrow=f[0].islower(), shape="elliptic", ellipse_spread=el, **flavors[f]
            ).scale_width(0.4)
        else:
            diagram.line(v[b], v[a], arrow=f[0].islower(), **flavors[f]).scale_width(0.4)

    diagram.plot()
