#
#    GPT - Grid Python Toolkit
#    Copyright (C) 2025  Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)
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


def checksums(data, grid, pos):
    site_crc = g.crc32(data, n=len(pos))
    global_site = grid.lexicographic_index(pos).astype(np.uint32)
    global_site_29 = np.mod(global_site, 29)
    global_site_31 = np.mod(global_site, 31)
    checksums_a = (site_crc << global_site_29) | (site_crc >> (32 - global_site_29))
    checksums_b = (site_crc << global_site_31) | (site_crc >> (32 - global_site_31))
    checksum_a = np.bitwise_xor.reduce(checksums_a)
    checksum_b = np.bitwise_xor.reduce(checksums_b)
    return checksum_a, checksum_b


def checksums_reduce(checksum_a, checksum_b, nreader, grid):
    crc_array = np.array([0] * (2 * nreader), np.uint64)
    if grid.processor < nreader:
        crc_array[2 * grid.processor + 0] = checksum_a
        crc_array[2 * grid.processor + 1] = checksum_b
    grid.globalsum(crc_array)
    crc_comp_a = 0x0
    crc_comp_b = 0x0
    for i in range(nreader):
        crc_comp_a ^= int(crc_array[2 * i + 0])
        crc_comp_b ^= int(crc_array[2 * i + 1])
    return crc_comp_a, crc_comp_b
