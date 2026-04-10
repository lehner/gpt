#!/usr/bin/env python3
import gpt as g
import numpy as np
import os

#g.default.set_verbose("io", False)
def ac(root, i0, i1, di, cdims):
    mf = {
        "E4" : [],
        "Q4" : [],
        "E8" : [],
        "Q8" : []
    }
    for i in range(i0, i1, di):
        g.message(i)
        U = g.load(root + "/ckpoint_lat." + str(i))
        for t in range(80):
            g.message(i,t)
            U = g.qcd.gauge.smear.wilson_flow(U, 0.1)
            if t == 39:
                mf["E4"].append(g.qcd.gauge.energy_density(U, field=True))
                mf["Q4"].append(g.qcd.gauge.topological_charge(U, field=True))
            elif t == 79:
                mf["E8"].append(g.qcd.gauge.energy_density(U, field=True))
                mf["Q8"].append(g.qcd.gauge.topological_charge(U, field=True))

    # deal with PBC edge effect
    zero = g.copy(mf["E4"][0])
    zero[:] = 0

    T0 = len(mf["E4"])
    for tg in mf:
        # first subtract mean, then padd with zeros
        mf_mean = sum(g.sum(x) for x in mf[tg]) / mf[tg][0].grid.fsites / len(mf[tg]) * g.identity(mf[tg][0])
        for x in mf[tg]:
            x -= mf_mean
        mf[tg] = mf[tg] + [zero]*T0
    correction = np.array([2*T0/(T0 - i) for i in range(T0)])

    T = len(mf["E4"])
    for x in mf:
        mf[x] = g.merge(mf[x])
        mf[x] = g.correlate(mf[x], mf[x], dims=[4])
        mf[x] = g.block.transfer(
            mf[x].grid, 
            g.grid(cdims + [T], g.double), 
            mf[x].otype
        ).sum(mf[x])

        mf[x] = mf[x][0:cdims[0],0:cdims[1],0:cdims[2],0:cdims[3],0:T].real.reshape(T, np.prod(cdims))


    if g.rank() == 0:
        if not os.path.exists(f"measures/{root}-ac2"):
            os.makedirs(f"measures/{root}-ac2", exist_ok=True)
    g.barrier()
    nblock = 3
    for i in range(0, np.prod(cdims), nblock):
        w = g.corr_io.writer(f"measures/{root}-ac2/{i//nblock}")
        for x in mf:
            w.write(x, np.mean([ mf[x][:,i+j] for j in range(nblock) ], axis=0)[0:T0] * correction)
        w.close()
    return mf

def aca(root, i0, i1, di=1):
    # 2,4,4,3 -> 16,8,8,16 volume per node
    i1 -= (i1 - i0) % 4
    ac(root, i0, i1, di, [2,4,4,6])

aca("xfthmc-0.124-therm-tau2", 0, 148)
aca("xfthmc-0.124-therm-tau4", 0, 112)
aca("xfthmc-0.124-therm-tau8", 0, 40)

aca("hmc-tau16", 64, 125)
aca("hmc-tau8", 125, 250)
aca("hmc-tau4", 250, 500)
aca("hmc-tau2", 500, 1000)
aca("hmc", 1000, 2000)
aca("hmc-tau0p5", 2800, 4000, 2)

g.barrier()
