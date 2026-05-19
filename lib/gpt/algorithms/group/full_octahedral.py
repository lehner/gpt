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


def orbit(mom):
    permutations = [(0, 1, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0), (0, 2, 1), (1, 0, 2)]
    mom = tuple(mom)
    assert len(mom) % 3 == 0
    orbit = set([])
    for px in [-1, 1]:
        for py in [-1, 1]:
            for pz in [-1, 1]:
                pi = (px, py, pz)
                for p in permutations:
                    o = tuple(pi[i % 3] * mom[p[i % 3] + (i // 3) * 3] for i in range(len(mom)))
                    orbit.add(o)
    return orbit


def from_fundamental(mom, i, f):
    assert len(mom) == 3
    o = orbit(mom)
    return sum(f(p, i) * g.algorithms.group.representation.operator(p) for p in o).normalized()


def t1u_l(mom, i, l):
    return from_fundamental(mom, i, lambda p, j: p[j] ** l)


def t1u_all(mom, i):
    l = 1
    ops = [t1u_l(mom, i, l)]
    while True:
        l += 2
        nop = t1u_l(mom, i, l)
        if g.algorithms.group.representation.redundant_operators(ops + [nop])[-1]:
            break
        ops = ops + [nop]
    return ops
