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


def matrix_vector(lat_matrix, lat_vector, points, code, code_parallel_block_size=None):
    # check if all points are cartesian
    for p in points:
        if len([s for s in p if s != 0]) > 1:
            raise Exception("General stencil matrix_vector not yet implemented")
    return g.local_stencil.matrix_vector(
        lat_matrix, lat_vector, points, code, code_parallel_block_size, local=0
    )
