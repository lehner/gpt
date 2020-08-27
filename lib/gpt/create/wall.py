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
import gpt
import numpy as np
import gpt.create.sparse_grid


def zn(src, t, n, rng, orthogonal=3):
    nd = src.grid.nd
    assert nd > orthogonal
    position = [0] * nd
    spacing = [1] * nd
    position[orthogonal] = t
    spacing[orthogonal] = np.iinfo(np.int32).max
    return gpt.create.sparse_grid.zn(src, position, spacing, rng, n)


def z2(src, t, rng, orthogonal=3):
    return zn(src, t, 2, rng, orthogonal)


def z3(src, t, rng, orthogonal=3):
    return zn(src, t, 3, rng, orthogonal)
