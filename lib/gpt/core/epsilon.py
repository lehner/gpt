#
#    GPT - Grid Python Toolkit
#    Copyright (C) 2022  Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)
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
import itertools as it

def permutation_sign(permutation):
    permutation = list(permutation)
    n = len(permutation)
    sign = 1.0
    for i in range(n-1):
        if permutation[i] != i:
            sign *= -1.0
            j = min(range(i,n), key=permutation.__getitem__)
            permutation[i], permutation[j] = permutation[j], permutation[i]
    return sign


def epsilon(n):
    return [(p, permutation_sign(p)) for p in it.permutations(range(n))]
