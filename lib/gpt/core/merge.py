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
import cgpt
import numpy
import sys

def merge(lattices,N = -1,dimension = -1):
    n                = len(lattices) ; assert(n>0)
    if N == -1:
        N            = n
    batches          = n // N ; assert(n % N == 0)
    grid             = lattices[0].grid  ; assert(all([ lattices[i].grid.obj == grid.obj for i in range(1,n) ]))
    if dimension < 0:
        dimension    = grid.nd + 1 + dimension
        assert(dimension >= 0)
    else:
        assert(dimension<=grid.nd)
    cb               = lattices[0].checkerboard() ; assert(all([ lattices[i].checkerboard() is cb for i in range(1,n) ]))
    otype            = lattices[0].otype ; assert(all([ lattices[i].otype.__name__ == otype.__name__ for i in range(1,n) ]))
    merged_grid      = grid.inserted_dimension(dimension,N)
    merged_lattices  = [ gpt.lattice(merged_grid,otype) for i in range(batches) ]
    gcoor            = gpt.coordinates( (grid,cb) )
    for i in range(N):
        merged_gcoor = cgpt.coordinates_inserted_dimension(gcoor,dimension,[ i ])
        for j in range(batches):
            merged_lattices[j][merged_gcoor] = lattices[j*N + i][gcoor]
    return merged_lattices

def separate(lattices,dimension = -1):
    batches            = len(lattices) ; assert(batches>0)
    grid               = lattices[0].grid  ; assert(all([ lattices[i].grid.obj == grid.obj for i in range(1,batches) ]))
    if dimension < 0:
        dimension      = grid.nd + dimension
        assert(dimension >= 0)
    else:
        assert(dimension<grid.nd)
    N                  = grid.fdimensions[dimension]
    n                  = N * batches
    cb                 = lattices[0].checkerboard() ; assert(all([ lattices[i].checkerboard() is cb for i in range(1,batches) ]))
    otype              = lattices[0].otype ; assert(all([ lattices[i].otype.__name__ == otype.__name__ for i in range(1,batches) ]))
    separated_grid     = grid.removed_dimension(dimension)
    separated_lattices = [ gpt.lattice(separated_grid,otype) for i in range(n) ]
    separated_gcoor    = gpt.coordinates( (separated_grid,cb) )
    for i in range(N):
        gcoor = cgpt.coordinates_inserted_dimension(separated_gcoor,dimension,[ i ])
        for j in range(batches):
            separated_lattices[j*N + i][separated_gcoor]=lattices[j][gcoor]
    return separated_lattices
