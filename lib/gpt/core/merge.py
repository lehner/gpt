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
    gpt.barrier()
    print("----")
    sys.stdout.flush()
    gpt.barrier()

    t0=gpt.time()
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
    t1=gpt.time()
    merged_grid      = grid.inserted_dimension(dimension,N)
    t2=gpt.time()
    merged_lattices  = [ gpt.lattice(merged_grid,otype) for i in range(batches) ]
    for x in merged_lattices:
        x.checkerboard(cb)
    t3=gpt.time()
    gcoor            = lattices[0].mview_coordinates() # return coordinates in internal ordering, speed up access
    t4=gpt.time()
    dt=0
    for i in range(N):
        dt-=gpt.time()
        merged_gcoor = cgpt.coordinates_inserted_dimension(gcoor,dimension,[ i ])
        dt+=gpt.time()
        gpt.poke(merged_lattices,merged_gcoor,gpt.peek([ lattices[j*N + i] for j in range(batches) ],gcoor))
    t5=gpt.time()
    if gpt.default.is_verbose("merge"):
        dataGB=grid.gsites * grid.precision.nbytes * otype.nfloats / grid.Nprocessors / 1024.**3. * 4.0 * n
        gpt.message("Merging at %g GB/s" % (dataGB/(t5-t0)),t1-t0,t2-t1,t3-t2,t4-t3,t5-t4,dt)
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
    for x in separated_lattices:
        x.checkerboard(cb)
    separated_gcoor    = separated_lattices[0].mview_coordinates()
    for i in range(N):
        gcoor = cgpt.coordinates_inserted_dimension(separated_gcoor,dimension,[ i ])
        gpt.poke([ separated_lattices[j*N + i] for j in range(batches) ],separated_gcoor,gpt.peek(lattices,gcoor))
    return separated_lattices
