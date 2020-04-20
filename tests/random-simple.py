#!/usr/bin/env python3
#
# Authors: Christoph Lehner 2020
#
# Desc.: Illustrate core concepts and features
#
import gpt as g

rng=g.random("block_seed_string_13")
n=10000
res={}
for i in range(n):
    z=rng.zn()
    if not z in res:
        res[z] = 0
    res[z] += 1

print(res)


n=10000
res={}
for i in range(n):
    z=rng.zn(p={ "n" : 3 })
    if not z in res:
        res[z] = 0
    res[z] += 1

print(res)


n=10000
res=[0,0,0,0,0]
for i in range(n):
    z=rng.normal()
    res[0]+=1
    res[1]+=z
    res[2]+=z**2
    res[3]+=z**3
    res[4]+=z**4

print(res[1] / res[0], res[2] / res[0], res[3] / res[0], res[4] / res[0])
