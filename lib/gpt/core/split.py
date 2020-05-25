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

def _separate_otypes(l):
    ot={}
    for i,x in enumerate(l):
        if not x.otype in ot:
            ot[x.otype]=[]
        ot[x.otype].append(i)
    return ot

# TODO: verbose split and time operations below

def split_lattices(lattices,lcoor,gcoor,split_grid,N):
    # N is desired number of parallel split lattices per unsplit lattice
    # 1 <= N <= sranks, sranks % N == 0
    assert(len(lcoor) == len(gcoor))
    n         = len(lattices) ; assert(n>0)
    grid      = lattices[0].grid  ; assert(all([ lattices[i].grid.obj == grid.obj for i in range(1,n) ]))
    cb        = lattices[0].checkerboard() ; assert(all([ lattices[i].checkerboard() is cb for i in range(1,n) ]))
    otype     = lattices[0].otype ; assert(all([ lattices[i].otype.__name__ == otype.__name__ for i in range(1,n) ]))
    l         = [ gpt.lattice(split_grid,otype) for i in range(n) ]
    batches   = n // N ; assert(n % N == 0)
    for i in range(n):
        l[i].checkerboard(cb)
    if N != 1:

        merged_lattices = merge_lattices(lattices,N)

        for i in range(batches):
            merged_lattices[i].checkerboard(cb)
            merged_lattices[i][merged_gcoor]

        for i in range(batches):
            lattices[i*N:(i+1)*N]

    gpt.poke(l,lcoor,gpt.peek(lattices,gcoor))
    for i in range(n):
        l[i].split_lcoor = lcoor
        l[i].split_gcoor = gcoor
    return l


def unsplit(first,second):
    if type(first) != list:
        return unsplit([ first ],[ second ])

    n            = len(first)
    N            = len(second)
    split_grid   = second[0].grid
    sranks       = split_grid.sranks
    srank        = split_grid.srank
    Q            = n // N ; assert ( n % N == 0)
    print(n,N,Q) ; sys.stdout.flush()

    # TODO: speed this up by first merging vectors in new 5d grid
    lcoor = second[0].split_lcoor
    gcoor = second[0].split_gcoor
    for i in range(Q):
        if i == srank // (sranks // Q):
            lc=lcoor
            gc=gcoor
        else:
            lc=numpy.empty(shape=(0,split_grid.nd),dtype=numpy.int32)
            gc=lc
        gpt.poke(first[i*N:(i+1)*N],gc,gpt.peek(second,lc))

def split_sites(first,sites,mpi_split = None):
    # this is the general case and should be used by the others
    # sites need to be the local sites we want to have on this rank
    # in the mpi_split layout
    #
    # split_by_rank : mpi_split = [1,1,1,1] ; sites = gpt.coordinates(first)
    # split_list    : mpi_split = param     ; sites = gpt.coordinates(first_on_split_grid)
    #
    # in general find compact envelope of sites (minimal ldimensions that can hold the sites)
    # this needs to be fast and should go into cgpt
    #
    return

def split_block(first,ldimensions,mpi_split):
    # this can still work with operators, block here
    pass

def split_by_rank(first):
    if type(first) != list:
        return split_by_rank([ first ])[0]

    assert(len(first) > 0)

    # TODO: split types
    lattices    = first
    grid        = lattices[0].grid
    mpi_split   = [1,1,1,1]
    fdimensions = [ grid.fdimensions[i] // grid.mpi[i] for i in range(grid.nd) ]
    split_grid  = grid.split(mpi_split,fdimensions)
    gcoor       = gpt.coordinates( lattices[0] )
    lcoor       = gpt.coordinates( (split_grid, lattices[0].checkerboard() ) )
    return split_lattices(lattices,lcoor,gcoor,split_grid,len(lattices))

def split(first, mpi_split = None):
    assert(len(first) > 0)
    lattices    = first
    grid        = lattices[0].grid
    split_grid  = grid.split(mpi_split)
    gcoor       = gpt.coordinates( (split_grid, lattices[0].checkerboard() ) )
    lcoor       = gpt.coordinates( (split_grid, lattices[0].checkerboard() ) )
    assert(len(lattices) % split_grid.sranks == 0)
    return split_lattices(lattices,lcoor,gcoor,split_grid,len(lattices) // split_grid.sranks)
