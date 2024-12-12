#
#    GPT - Grid Python Toolkit
#    Copyright (C) 2024  Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)
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


def angle(w):
    # Heron's method
    uk = g(w / g.norm2(w) ** 0.5 * 1e3)
    I = g.identity(w)
    nrm = g.norm2(I)
    for i in range(20):
        uk = g(0.5 * uk + g.matrix.inv(g.adj(uk)) * 0.5)
        err2 = g.norm2(uk * g.adj(uk) - I) / nrm
        if err2 < w.grid.precision.eps**2 * 10:
            return uk
    raise Exception("angle did not converge")


def decompose(w):
    u = angle(w)
    h = g(w * g.adj(u))
    err2 = g.norm2(h - g.adj(h)) / g.norm2(h)
    assert err2 < w.grid.precision.eps**2 * 100
    return h, u
