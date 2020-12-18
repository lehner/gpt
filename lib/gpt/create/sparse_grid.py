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


def coordinates(src, position, spacing):
    coor = gpt.coordinates(src)
    return coor[np.sum(np.mod(coor - position, spacing), axis=1) == 0]


def zn(src, position, spacing, rng, n):
    singlet = gpt.lattice(src.grid, gpt.ot_singlet)
    singlet.checkerboard(src.checkerboard())
    pos = coordinates(src, position, spacing)
    singlet_full = gpt.lattice(singlet)
    rng.zn(singlet_full, n=n)
    singlet[:] = 0
    singlet[pos] = singlet_full[pos]
    src @= src.otype.identity() * singlet
    return src
