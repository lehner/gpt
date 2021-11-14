#!/usr/bin/env python3
#
# Authors: Christoph Lehner 2020
#
# Calculate HVP connected diagram with A2A method
#
import gpt as g
import numpy as np
import sys, os

# configure
root_output = "/pscratch/sd/l/lehner/distillation"

groups = {
    "pm_batch_0": {
        "confs": [ #"440", "480", "520", "560", "600", "640", "680", "720", "760", "800", "840", "880", "920", "1000", 
            "960", "1040"
        ],
        "evec_fmt": "/global/cfs/projectdirs/m3886/lehner/%s/lanczos.output",
        "conf_fmt": "/global/cfs/projectdirs/mp13/lehner/96I/ckpoint_lat.%s",
        "basis_fmt":"/pscratch/sd/l/lehner/distillation/%s_basis/basis_t%d"
    },
}

jobs = {}

sloppy_per_job = 50
basis_size = 200

for t in [0, 2]: #range(0, 192, 2)
    for i0 in range(0, basis_size, sloppy_per_job):
        jobs[f"pm_sloppy_t{t}_i{i0}"] = {
            "solver" : "sloppy",
            "t" : t,
            "i" : list(range(i0, i0+sloppy_per_job))
        }

jobs_per_run = g.default.get_int("--gpt_jobs", 1)

# find jobs for this run
def get_job(only_on_conf=None):
    # statistics
    n = 0
    for group in groups:
        for job in jobs:
            for conf in groups[group]["confs"]:
                n += 1

    jid = -1
    # for job in jobs:
    #    for group in groups:
    #        for conf in groups[group]["confs"]:
    for group in groups:
        for conf in groups[group]["confs"]:
            for job in jobs:
                jid += 1
                if only_on_conf is not None and only_on_conf != conf:
                    continue
                root_job = f"{root_output}/{conf}/{job}"
                if not os.path.exists(root_job):
                    os.makedirs(root_job)
                    return group, job, conf, jid, n

    return None, None, None, None, None


if g.rank() == 0:
    first_job = get_job()
    run_jobs = str(
        list(
            filter(
                lambda x: x is not None,
                [first_job] + [get_job(first_job[2]) for i in range(1, jobs_per_run)],
            )
        )
    ).encode("utf-8")
else:
    run_jobs = bytes()
run_jobs = eval(g.broadcast(0, run_jobs).decode("utf-8"))

# every node now knows what to do
g.message(
    """
================================================================================
       Distillation perambulator run ;  this run will attempt:
================================================================================
"""
)
for group, job, conf, jid, n in run_jobs:
    if group is None:
        break
    g.message(
        f"""

    Job {jid} / {n} :  configuration {conf}, job tag {job}

"""
    )

if len(run_jobs) == 0:
    sys.exit(0)
    
# configuration needs to be the same for all jobs, so load eigenvectors and configuration
conf = run_jobs[0][2]
group = run_jobs[0][0]

U = g.load(groups[group]["conf_fmt"] % conf)
L = U[0].grid.fdimensions

l_exact = g.qcd.fermion.mobius(
    U,
    {
        "mass": 0.00054,
        "M5": 1.8,
        "b": 1.5,
        "c": 0.5,
        "Ls": 12,
        "boundary_phases": [1.0, 1.0, 1.0, -1.0],
    },
)

l_sloppy = l_exact.converted(g.single)

eig = g.load(groups[group]["evec_fmt"] % conf, grids=l_sloppy.F_grid_eo)

# pin coarse eigenvectors to GPU memory
pin = g.pin(eig[1], g.accelerator)

light_innerL_inverter = g.algorithms.inverter.preconditioned(
    g.qcd.fermion.preconditioner.eo1_ne(parity=g.odd),
    g.algorithms.inverter.sequence(
        g.algorithms.inverter.coarse_deflate(
            eig[1],
            eig[0],
            eig[2],
            block=400,
            fine_block=4,
            linear_combination_block=32,
        ),
        g.algorithms.inverter.split(
            g.algorithms.inverter.cg({"eps": 1e-8, "maxiter": 200}),
            mpi_split=g.default.get_ivec("--mpi_split", None, 4),
        ),
    ),
)

light_innerH_inverter = g.algorithms.inverter.preconditioned(
    g.qcd.fermion.preconditioner.eo1_ne(parity=g.odd),
    g.algorithms.inverter.sequence(
        g.algorithms.inverter.coarse_deflate(
            eig[1],
            eig[0],
            eig[2],
            block=400,
            fine_block=4,
            linear_combination_block=32,
        ),
        g.algorithms.inverter.split(
            g.algorithms.inverter.cg({"eps": 1e-8, "maxiter": 300}),
            mpi_split=g.default.get_ivec("--mpi_split", None, 4),
        ),
    ),
)

light_low_inverter = g.algorithms.inverter.preconditioned(
    g.qcd.fermion.preconditioner.eo1_ne(parity=g.odd),
    g.algorithms.inverter.coarse_deflate(
        eig[1], eig[0], eig[2], block=400, linear_combination_block=32, fine_block=4
    ),
)

light_exact_inverter = g.algorithms.inverter.defect_correcting(
    g.algorithms.inverter.mixed_precision(light_innerH_inverter, g.single, g.double),
    eps=1e-8,
    maxiter=10,
)

light_sloppy_inverter = g.algorithms.inverter.defect_correcting(
    g.algorithms.inverter.mixed_precision(light_innerL_inverter, g.single, g.double),
    eps=1e-8,
    maxiter=2,
)

prop_l_low = l_sloppy.propagator(light_low_inverter)
prop_l_sloppy = l_exact.propagator(light_sloppy_inverter).grouped(2)
prop_l_exact = l_exact.propagator(light_exact_inverter).grouped(2)

# show available memory
g.mem_report(details=False)

# per job
for group, job, conf, jid, n in run_jobs:
    if group is None:
        break
    g.message(
        f"""

    Job {jid} / {n} :  configuration {conf}, job tag {job}

"""
    )

    t = jobs[job]["t"]
    ilist = jobs[job]["i"]
    solver = jobs[job]["solver"]
    prop_l = {
        "sloppy" : prop_l_sloppy,
        "exact" : prop_l_exact
    }[solver]
    
    basis_evec, basis_evals = g.load(groups[group]["basis_fmt"] % (conf,t))

    g.message(f"t = {t}, ilist = {ilist}, basis size = {len(basis_evec)}, solver = {solver}")

    root_job = f"{root_output}/{conf}/{job}"
    output = g.gpt_io.writer(f"{root_job}/propagators")
    
    # create sources
    srcD = [g.vspincolor(l_exact.U_grid) for spin in range(4)]

    for i in ilist:
        
        c = g.coordinates(basis_evec[i])

        for spin in range(4):
            srcD[spin][:] = 0
            srcD[spin][np.hstack((c, np.ones((len(c),1),dtype=np.int32)*t)),spin,:] = basis_evec[i][c]

            g.message("Norm of source:",g.norm2(srcD[spin]))
            
        prop = g.eval(prop_l * srcD)
        g.mem_report(details=False)

        for spin in range(4):
            output.write({f"t{t}s{spin}c{i}_{solver}": prop[spin]})
            output.flush()

del pin
