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
import cgpt, gpt
import numpy as np

# memoryview(data) is unnecessarily slow, cgpt version is faster
def mview(data):
    if type(data) == memoryview:
        return data
    return cgpt.mview(data)


# fast threaded checksum of memoryviews
def crc32(view, crc32_prev=0):
    if type(view) == memoryview:
        return cgpt.util_crc32(view, crc32_prev)
    else:
        return crc32(memoryview(view), crc32_prev)


# distribute loading of cartesian file with lexicographic ordering
def distribute_cartesian_file(fdimensions, grid, cb):
    ldimensions = [x for x in fdimensions]
    dimdiv = len(ldimensions) - 1
    primes = [7, 5, 3, 2]
    nreader = 1
    found = True
    while found:
        found = False
        for p in primes:
            if ldimensions[dimdiv] % p == 0 and nreader * p <= grid.Nprocessors:
                nreader *= p
                ldimensions[dimdiv] //= p
                if ldimensions[dimdiv] == 1 and dimdiv > 0:
                    dimdiv -= 1
                found = True

    cv_desc = [a // b for a, b in zip(fdimensions, ldimensions)]
    cv = gpt.cartesian_view(grid.processor, cv_desc, fdimensions, grid.cb, cb)
    return gpt.coordinates(cv), nreader
