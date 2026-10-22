#
#    GPT - Grid Python Toolkit
#    Copyright (C) 2026  Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)
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

# --lattice_pool: keep the cgpt objects of deleted lattices and reuse them
# for new lattices of the same grid, component type, and precision, so that
# no field memory is freed or allocated after the first use of a shape.
# cgpt never deletes a grid object and returns the same one for the same
# geometry, so pooled lattices stay valid and are shared by recreated grids.
use_pool = gpt.default.has("--lattice_pool")
pool = {}


def create_lattice_obj(grid, t):
    if use_pool:
        free = pool.get((grid.obj, t, grid.precision.cgpt_dtype))
        if free:
            o = free.pop()
            if grid.cb.n != 1:
                # a new cgpt lattice starts on the even checkerboard
                cgpt.lattice_change_checkerboard(o, gpt.even.tag)
            return o
    return cgpt.create_lattice(grid.obj, t, grid.precision.cgpt_dtype)


def delete_lattice_obj(grid, t, o):
    if use_pool:
        pool.setdefault((grid.obj, t, grid.precision.cgpt_dtype), []).append(o)
    else:
        cgpt.delete_lattice(o)


def release_pool():
    for free in pool.values():
        for o in free:
            cgpt.delete_lattice(o)
    pool.clear()
