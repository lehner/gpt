#!/usr/bin/env python3
#
# Authors: Christoph Lehner 2020
#
# Desc.: Illustrate core concepts and features
#
import gpt as g
import numpy as np

rng=g.random("block_seed_string_13")
n=10000
res={}
for i in range(n):
    z=rng.zn()
    if not z in res:
        res[z] = 0
    res[z] += 1

g.message(res)


n=10000
res={}
for i in range(n):
    z=rng.zn(p={ "n" : 3 })
    if not z in res:
        res[z] = 0
    res[z] += 1

g.message(res)


n=10000
res=[0,0,0,0,0]
for i in range(n):
    z=rng.normal()
    res[0]+=1
    res[1]+=z
    res[2]+=z**2
    res[3]+=z**3
    res[4]+=z**4

g.message(res[1] / res[0], res[2] / res[0], res[3] / res[0], res[4] / res[0])

grid=g.grid([4,4,4,4],g.double)
v=g.complex(grid)
rng.normal(v)

test_sequence_comp=np.array([ v[0,0,0,0].real, v[2,0,0,0].real, v[0,2,0,0].real, v[1,3,1,3].real, v[3,2,1,0].real ])
#print([ v[0,0,0,0].real, v[2,0,0,0].real, v[0,2,0,0].real, v[1,3,1,3].real, v[3,2,1,0].real ])
test_sequence_ref =np.array([ 1.0336693180495347, -0.23474901515559715, -0.26622475825072717, 1.0175124089453662, 0.9519248978004753 ],np.float64)

g.message(test_sequence_comp)

err=np.linalg.norm(test_sequence_comp - test_sequence_ref)
assert(err < 1e-14)

g.message(err)

for i in range(1000):
    rng.normal(v)

test_sequence_comp=np.array([ v[0,0,0,0].real, v[2,0,0,0].real, v[0,2,0,0].real, v[1,3,1,3].real, v[3,2,1,0].real ])
#print([ v[0,0,0,0].real, v[2,0,0,0].real, v[0,2,0,0].real, v[1,3,1,3].real, v[3,2,1,0].real ])
test_sequence_ref =np.array([-0.3194228736274878, -0.4211727874061459, 0.7680875563790006, -0.18697640758578687, 0.3276024231946795],np.float64)
g.message(test_sequence_comp)

err=np.linalg.norm(test_sequence_comp - test_sequence_ref)
assert(err < 1e-14)

g.message("All tests passed")
