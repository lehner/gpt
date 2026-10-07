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


# The kind of a matrix stencil follows from its points: points on the axes
# only and no temporaries -> Grid's cartesian stencil (comm_type=0); any
# other points, or kernel-owned temporaries -> the general stencil
# (comm_type=2: a halo exchange into a comm buffer, see
# lib/cgpt/lib/foundation/general_stencil.h), which supports full grids
# only.  (The former padded stencil for checkerboarded grids was removed
# 2026-10-07: nothing used it.)
def matrix(lat, points, code, code_parallel_block_size=None, temporaries=(), cse=False):
    cartesian = all(len([s for s in p if s != 0]) <= 1 for p in points)
    if cartesian and len(temporaries) == 0:
        # (a cartesian kernel runs without temporaries: no cse)
        return g.local_stencil.matrix(lat, points, code, code_parallel_block_size, comm_type=0)
    if lat.grid.cb.n != 1:
        raise NotImplementedError(
            "g.stencil.matrix: points off the axes (or local temporaries) need the general "
            "stencil, which supports full grids only; this lattice is checkerboarded"
        )
    return g.local_stencil.matrix(
        lat, points, code, code_parallel_block_size, comm_type=2, temporaries=temporaries, cse=cse
    )
