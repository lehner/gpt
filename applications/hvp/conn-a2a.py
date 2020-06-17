#!/usr/bin/env python3
#
# Authors: Christoph Lehner 2020
#
# Calculate HVP connected diagram with A2A method
#
import gpt as g
import numpy as np
import sys

# show available memory
g.mem_report()

# parameters
fn=g.default.get("--params","params.txt")
params=g.params(fn,verbose = True)

# load configuration
U = params["config"]
L = U[0].grid.fdimensions

# show available memory
g.mem_report()

# quark flavors
l_exact=params["light_exact"](U)
l_sloppy=params["light_sloppy"](U,l_exact)

# eigenvectors and values, may be MG
l_sloppy_eigensystem=params["light_sloppy_eigensystem"](U,l_sloppy)

# show available memory
g.mem_report()

# sloppy solver
prop_l_sloppy=params["light_propagator_sloppy"](U,l_sloppy,l_sloppy_eigensystem)
params["light_propagator_sloppy_test"](prop_l_sloppy)

# exact solver
prop_l_exact=params["light_propagator_exact"](U,l_exact,prop_l_sloppy)

# random number generator
rng=params["rng"]

# AMA strategy
# (exact - sloppy)_positions_exact + (sloppy - low)_positions_sloppy + low_volume_average

# source positions
source_positions_sloppy=params["source_positions_sloppy"](rng,L)
source_positions_exact=params["source_positions_exact"](rng,L)

# output
output=params["output"]
output_correlator=params["output_correlator"]

# save positions
pos_archive={ 
    "source_positions_sloppy" : source_positions_sloppy,
    "source_positions_exact" : source_positions_exact
}
output.write(pos_archive)
g.message("Source Positions:")
g.message(pos_archive)

# correlators
operators=params["operators"]
correlators=params["correlators"]

def contract(pos,prop,tag):
    t0=pos[3]
    prop_tag="%s/%s" % (tag,str(pos))

    # save propagators
    if params["save_propagators"]:
        output.write({ prop_tag : prop })
        output.flush()
    
    # create and save correlators
    for op_snk,op_src in correlators:
        G_snk=operators[op_snk]
        G_src=operators[op_src]
        corr=g.slice(g.trace(G_src*g.gamma[5]*g.adj(prop)*g.gamma[5]*G_snk*prop),3)
        corr=corr[t0:] + corr[:t0]

        corr_tag="%s/snk%s-src%s" % (prop_tag,op_snk,op_src)
        output_correlator.write(corr_tag,corr)
        g.message("Correlator %s\n" % corr_tag,corr)

# calculate correlators for exact positions
for pos in source_positions_exact:
    
    # exact_sloppy
    src=g.mspincolor(l_sloppy.U_grid)
    g.create.point(src, pos)
    contract(pos, g.eval( prop_l_sloppy * src ), "sloppy" )

    # exact
    src=g.mspincolor(l_exact.U_grid)
    g.create.point(src, pos)
    contract(pos, g.eval( prop_l_exact * src ), "exact" )

# calculate correlators for sloppy positions
for pos in source_positions_sloppy:
    
    # sloppy
    src=g.mspincolor(l_sloppy.U_grid)
    g.create.point(src, pos)
    contract(pos, g.eval( prop_l_sloppy * src ), "sloppy" )

# create a2a vectors


# cleanup
params=None
output.close()
sys.exit(0)
