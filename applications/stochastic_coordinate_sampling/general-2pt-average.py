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
cout = corr_io.writer(f"general-2pt-average.{conf}")

# establish tags
tags = [t.split("/") for t in cin.tags_with_duplicates]
tags = [(t[0],t[1],t[2],"/".join(t[3:])) for t in tags]
keys = list(set( (t[1],t[3]) for t in tags ))

# positions for AMA correction
pos_ems = list( t[2] for t in tags if t[0] == "ems" and t[1] == keys[0][0] and t[3] == keys[0][1])

# positions for SLOPPY solves
pos_s = list( t[2] for t in tags if t[0] == "s" and t[1] == keys[0][0] and t[3] == keys[0][1])

# statistics
print(f"Computing average for {len(keys)} correlators averaged over {len(pos_ems)} AMA positions and {len(pos_s)} SLOPPY positions")

# positions for SLOPPY solves
for key in keys:
    fmt_key = "/".join(key)
    print(f"Processing", fmt_key)
    ems = np.mean([ cin.tags["/".join(["ems", key[0], p, key[1]])] for p in pos_ems], axis=0)
    s = np.mean([ cin.tags["/".join(["s", key[0], p, key[1]])] for p in pos_s], axis=0)
    cout.write(fmt_key, ems + s)
