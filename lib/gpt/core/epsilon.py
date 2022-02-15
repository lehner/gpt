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


def sign_of_permutation(permutation, reference):
    permutation = list(permutation)
    n = len(permutation)
    sign = 1.0
    for i in range(n - 1):
        if permutation[i] != reference[i]:
            sign *= -1.0
            j = permutation.index(reference[i])
            permutation[i], permutation[j] = permutation[j], permutation[i]
    for i in range(n):
        assert permutation[i] == reference[i]
    return sign


def epsilon(n):
    reference = list(range(n))
    return [(p, sign_of_permutation(p, reference)) for p in it.permutations(reference)]
