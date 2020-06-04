#!/usr/bin/env python3
#
# Authors: Christoph Lehner 2020
#
import gpt as g
import numpy as np
import sys

rng=g.random("test")
vol=[16,16,16,32]
grid_rb=g.grid(vol,g.single,g.redblack)
grid=g.grid(vol,g.single)
field=g.vcolor

################################################################################
# Setup lattices
################################################################################
l_rb=[ field(grid_rb) for i in range(8) ]
l=[ field(grid) for i in range(8) ]
for i in range(8):
    l_rb[i].checkerboard(g.odd)
rng.cnormal(l_rb)
rng.cnormal(l)

################################################################################
# Reverse lattice in time order in two ways and cross-check
################################################################################
def lattice_reverse_check(lat):
    grid=lat.grid
    l_rev_ref=g.lattice(lat)
    l_rev_ref.checkerboard(lat.checkerboard().inv())
    T=grid.fdimensions[3]
    for t in range(T):
        # 31 <- 0 => interchange even and odd sites
        l_rev_ref[:,:,:,T-t-1]=lat[:,:,:,t]

    l_rev=g.merge(list(reversed(g.separate(lat,3))),3)
    eps=g.norm2(l_rev - l_rev_ref)
    g.message("Temporal inverse lattice test: ",eps)
    assert(eps==0.0)

lattice_reverse_check(l_rb[0])
lattice_reverse_check(l[0])

################################################################################
# Test merge/separate here
################################################################################
assert(all([ g.norm2(x) > 0 for x in l ]))

# Test merging slices along a new last dimension 4 at a time
m=g.merge(l,N=4)
assert(len(m) == 2)

for i in range(len(m)):
    for j in range(4):
        k = i*4 + j
        assert(g.norm2(l[k][1,2,0,0] - m[i][1,2,0,0,j]) == 0.0)

# Test merging slices along a new 2nd dimension 4 at a time
m=g.merge(l,1,N=4)
assert(len(m) == 2)

for i in range(len(m)):
    for j in range(4):
        k = i*4 + j
        assert(g.norm2(l[k][1,2,0,0] - m[i][1,j,2,0,0]) == 0.0)

test=g.separate(m,1)

assert(len(test) == 8)
for i in range(len(l)):
    assert(g.norm2(l[i] - test[i]) == 0.0)

# default arguments should be compatible
test=g.separate(g.merge(l))
for i in range(len(l)):
    assert(g.norm2(l[i] - test[i]) == 0.0)

sys.exit(0)
################################################################################
# multi-vector (as in right-hand sides) splitting
# test split CG both speed and correctness against original
################################################################################



test=[ g.complex(grid) for i in range(8) ]
l_rank = g.split(l,mpi_split=[1,1,2,2]) # now is a list of lattices to deal with locally
g.barrier()

g.message("Local workload: ",len(l_rank))
g.unsplit(test,l_rank)
for i in range(len(test)):
    assert(g.norm2(test[i] - l[i]) == 0.0)


################################################################################
# split by ranks
################################################################################
l=g.complex(grid)
rng.cnormal(l)
l_rank = g.split_by_rank(l)

assert(l_rank.grid.globalsum(1.0) == 1.0) # check separate mpi grid

test=g.lattice(l)
test[:]=0
g.unsplit(test,l_rank)

assert(g.norm2(test - l) == 0.0)


# split many at same time (faster since it shares coordinates and grid creation); also can combine different lattice types
l=[ g.vcolor(grid) for i in range(8) ]
rng.cnormal(l)
l_rank=g.split_by_rank(l)

for i in l_rank:
    assert(i.grid.globalsum(1.0) == 1.0) # check that they live in separate mpi grid

test=[ g.vcolor(grid) for i in range(8) ]

g.unsplit(test,l_rank)

for i in range(len(test)):
    assert(g.norm2(test[i] - l[i]) == 0.0)


################################################################################
# split sites, useful for sub-blocking AND for sparse lattices with random 
# points
################################################################################

#for iblock:
#    pos_of_block=g.coordinates_of_block(iblock)
#    l_block = g.split_sites(l_rank,pos_of_block)
#    g.unsplit(l_rank,l_block)
#    l_block[g.boundary_points(...)]=0
