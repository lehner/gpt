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
#    Authors:  Lorenzo Barca     2020
#              Christoph Lehner  2020
#
import gpt as g
from gpt.core.colormatr import quarkContractXX

def quarkContract12(mspincolor1 , mspincolor2):

    grid = mspincolor1.grid
    dst = g.mspincolor(grid)
    dst[:] = 0
    mspin = g.separate_spin(dst)
    mspin1 = g.separate_spin(mspincolor1)
    mspin2 = g.separate_spin(mspincolor2)

    for s1 in range(4):
        for s2 in range(4):
            for k in range(4):
                mspin[s1, s2] += quarkContractXX(mspin1[k, k], mspin2[s1, s2], grid)

    g.merge_spin(dst, mspin)
    return dst


def quarkContract13(mspincolor1 , mspincolor2):

    grid = mspincolor1.grid
    dst = g.mspincolor(grid)
    dst[:] = 0
    mspin = g.separate_spin(dst)
    mspin1 = g.separate_spin(mspincolor1)
    mspin2 = g.separate_spin(mspincolor2)

    for s1 in range(4):
        for s2 in range(4):
            for k in range(4):
                mspin[s1, s2] += quarkContractXX(mspin1[k, s1], mspin2[k, s2], grid)

    g.merge_spin(dst, mspin)
    return dst


def quarkContract14(mspincolor1 , mspincolor2):

    grid = mspincolor1.grid
    dst = g.mspincolor(grid)
    dst[:] = 0
    mspin = g.separate_spin(dst)
    mspin1 = g.separate_spin(mspincolor1)
    mspin2 = g.separate_spin(mspincolor2)

    for s1 in range(4):
        for s2 in range(4):
            for k in range(4):
                mspin[s1, s2] += quarkContractXX(mspin1[k, s1], mspin2[s2, k], grid)

    g.merge_spin(dst, mspin)
    return dst


def quarkContract23(mspincolor1 , mspincolor2):

    grid = mspincolor1.grid
    dst = g.mspincolor(grid)
    dst[:] = 0
    mspin = g.separate_spin(dst)
    mspin1 = g.separate_spin(mspincolor1)
    mspin2 = g.separate_spin(mspincolor2)

    for s1 in range(4):
        for s2 in range(4):
            for k in range(4):
                mspin[s1, s2] += quarkContractXX(mspin1[s1, k], mspin2[k, s2], grid)

    g.merge_spin(dst, mspin)
    return dst


def quarkContract34(mspincolor1 , mspincolor2):

    grid = mspincolor1.grid
    dst = g.mspincolor(grid)
    dst[:] = 0
    mspin = g.separate_spin(dst)
    mspin1 = g.separate_spin(mspincolor1)
    mspin2 = g.separate_spin(mspincolor2)

    for s1 in range(4):
        for s2 in range(4):
            for k in range(4):
                mspin[s1, s2] += quarkContractXX(mspin1[s1, s2], mspin2[k, k], grid)

    g.merge_spin(dst, mspin)
    return dst
