#
#    GPT - Grid Python Toolkit
#    Copyright (C) 2023  Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)
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


def inner_product(sx, sy, use_accelerator):
    assert len(sx) == 1 and len(sy) == 1
    sx = sx[0]
    sy = sy[0]
    return {(0, 0): sx.distribute2(sy, lambda a, b: g.inner_product(a, b, use_accelerator))}


def norm2(sx):
    assert len(sx) == 1
    return [inner_product(sx, sx, True)[0, 0]]


def cshift(sx, mu, disp, none=None):
    assert none is None
    return sx.distribute1(lambda a: g.cshift(a, mu, disp))


def trace(sx, t):
    return sx.distribute1(lambda a: g.trace(a, t))


def adj(sx):
    return sx.distribute1(lambda a: g.adj(a))


def sum(sx):
    return sx.distribute1(lambda a: g.sum(a))
