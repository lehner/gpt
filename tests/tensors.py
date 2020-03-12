#!/usr/bin/env python3
#
# Authors: Christoph Lehner 2020
#
# Desc.: Illustrate core concepts and features
#
import gpt as g
import numpy as np

# gpt messaging system, only prints on g.rank() == 0
g.message("Using grid: ", g.default.grid)

# create a single precision grid with dimensions taken from "--grid ..."
grid=g.grid(g.default.grid, g.single)

cm=g.mcolor(grid)
cv=g.vcolor(grid)

cv @= cm * cv

cm @= g.adj(cv) * cv

