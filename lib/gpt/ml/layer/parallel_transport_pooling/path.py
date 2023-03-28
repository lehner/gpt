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
import numpy as np


def lexicographic(coordinate):
    path = g.path()
    for i in range(len(coordinate)):
        path.f(i, int(coordinate[i]))
    return path


def one_step_lexicographic(coordinate):
    path = g.path()
    nleft = np.array([c for c in coordinate], dtype=np.int32)
    while not np.all(nleft == 0):
        for i in range(len(coordinate)):
            if nleft[i] > 0:
                path.f(i, 1)
                nleft[i] -= 1
            elif nleft[i] < 0:
                path.b(i, 1)
                nleft[i] += 1
    return path


def reversed_lexicographic(coordinate):
    path = g.path()
    for i in reversed(range(len(coordinate))):
        path.f(i, int(coordinate[i]))
    return path


def one_step_reversed_lexicographic(coordinate):
    path = g.path()
    nleft = np.array([c for c in coordinate], dtype=np.int32)
    while not np.all(nleft == 0):
        for i in reversed(range(len(coordinate))):
            if nleft[i] > 0:
                path.f(i, 1)
                nleft[i] -= 1
            elif nleft[i] < 0:
                path.b(i, 1)
                nleft[i] += 1
    return path
