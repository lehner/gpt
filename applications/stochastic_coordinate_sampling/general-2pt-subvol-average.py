#!/usr/bin/env python3
#
# Content: general 2pt mesonic contractions from sparse point source data sets
# Author: Christoph Lehner
# Date: March 2026
#
import corr_io
import numpy as np
import sys

conf = int(sys.argv[1])

cin = corr_io.reader(f"general-2pt.{conf}")

# establish tags
tags = [t.split("/") for t in cin.tags_with_duplicates]
tags = [(t[0],t[1],t[2],"/".join(t[3:])) for t in tags]
keys = list(set( (t[1],t[3]) for t in tags ))

def sort_key(p, mu):
    assert isinstance(p, str)
    return int(p[1:-1].strip().replace('   ',' ').replace('  ',' ').split(' ')[mu])

def partition_dimension(pos, mu, res):
    if mu == 4:
        res.append(pos)
        return
    pos_mu = sorted(pos, key=lambda p: sort_key(p, mu))
    print(pos_mu)
    first = pos_mu[0:len(pos_mu)//2]
    second = pos_mu[len(pos_mu)//2:]
    assert len(first) == len(second)
    partition_dimension(first, mu+1, res)
    partition_dimension(second, mu+1, res)

def partition(pos):
    assert len(pos) % 16 == 0
    res = []
    partition_dimension(pos, 0, res)
    return res

# positions for AMA correction
pos_ems = partition(list( t[2] for t in tags if t[0] == "ems" and t[1] == keys[0][0] and t[3] == keys[0][1]))
for x in pos_ems:
    print(len(x))

# positions for SLOPPY solves
pos_s = partition(list( t[2] for t in tags if t[0] == "s" and t[1] == keys[0][0] and t[3] == keys[0][1]))

# statistics
print(f"Computing average for {len(keys)} correlators averaged over {len(pos_ems)} AMA positions and {len(pos_s)} SLOPPY positions")

for sv in range(16):

    cout = corr_io.writer(f"general-2pt-average.{conf}{sv:02d}")

    for key in keys:
        fmt_key = "/".join(key)
        print(f"Processing", fmt_key)

        ems = np.mean([ cin.tags["/".join(["ems", key[0], p, key[1]])] for p in pos_ems[sv]], axis=0)
        s = np.mean([ cin.tags["/".join(["s", key[0], p, key[1]])] for p in pos_s[sv]], axis=0)
        cout.write(fmt_key, ems + s)
