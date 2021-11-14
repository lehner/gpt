#!/usr/bin/env python3
#
# Authors: Christoph Lehner 2020
#
# Calculate HVP connected diagram with A2A method
#
import gpt as g
import numpy as np
import sys, os, glob


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

jobs = []

propagator_nodes_check = 1024
sloppy_per_job = 50
basis_size = 200

current_config = None
current_light_quark = None

class config:
    def __init__(self, conf_file):
        self.conf_file = conf_file
        self.U = g.load(conf_file)
        self.L = self.U[0].grid.fdimensions

        self.l_exact = g.qcd.fermion.mobius(
            self.U,
            {
                "mass": 0.00054,
                "M5": 1.8,
                "b": 1.5,
                "c": 0.5,
                "Ls": 12,
                "boundary_phases": [1.0, 1.0, 1.0, -1.0],
            },
        )

        self.l_sloppy = self.l_exact.converted(g.single)

class light_quark:
    def __init__(self, config, evec_dir):
        self.evec_dir = evec_dir
        self.eig = g.load(evec_dir, grids=config.l_sloppy.F_grid_eo)

        g.mem_report(details = False)

        # pin coarse eigenvectors to GPU memory
        self.pin = g.pin(self.eig[1], g.accelerator)

        light_innerL_inverter = g.algorithms.inverter.preconditioned(
            g.qcd.fermion.preconditioner.eo1_ne(parity=g.odd),
            g.algorithms.inverter.sequence(
                g.algorithms.inverter.coarse_deflate(
                    self.eig[1],
                    self.eig[0],
                    self.eig[2],
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
                    self.eig[1],
                    self.eig[0],
                    self.eig[2],
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
                self.eig[1], self.eig[0], self.eig[2], block=400, linear_combination_block=32, fine_block=4
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
    
        self.prop_l_low = l_sloppy.propagator(light_low_inverter)
        self.prop_l_sloppy = l_exact.propagator(light_sloppy_inverter).grouped(2)
        self.prop_l_exact = l_exact.propagator(light_exact_inverter).grouped(2)


def propagator_consistency_check(dn, n0):
    fs = glob.glob(f"{dn}/??/*.field")
    n = len(fs)
    if n != n0:
        return False
    s0 = os.path.getsize(fs[0])
    if s0 == 0:
        return False
    return all([ s0 == os.path.getsize(f) for f in fs[1:] ])


class job_perambulator(g.jobs.base):
    def __init__(self, conf, evec_dir, conf_file, basis_dir, t, i0, solver):
        self.conf = conf
        self.solver = solver
        self.conf_file = conf_file
        self.evec_dir = evec_dir
        self.basis_dir = basis_dir
        self.t = t
        self.i0 = i0
        self.ilist = list(range(i0, i0+sloppy_per_job))
        super().__init__(f"{conf}/pm_{solver}_t{t}_i{i0}",[])

    def perform(self, root):
        global current_config, current_light_quark
        if current_config is not None and current_config.conf_file != self.conf_file:
            current_config = None
        if current_config is None:
            current_config = config(self.conf_file)

        if current_light_quark is not None and current_light_quark.evec_dir != self.evec_dir:
            current_light_quark = None
        if current_light_quark is None:
            current_light_quark = light_quark(current_config, self.evec_dir)

        prop_l = {
            "sloppy" : current_light_quark.prop_l_sloppy,
            "exact" : current_light_quark.prop_l_exact
        }[self.solver]
        
        basis_evec, basis_evals = g.load(self.basis_dir)

        g.message(f"t = {self.t}, ilist = {self.ilist}, basis size = {len(basis_evec)}, solver = {self.solver}")

        root_job = f"{root}/{self.name}"
        output = g.gpt_io.writer(f"{root_job}/propagators")
    
        # create sources
        srcD = [g.vspincolor(current_config.l_exact.U_grid) for spin in range(4)]

        for i in self.ilist:
        
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

    def check(self, root):
        return propagator_consistency_check(f"{root}/{self.name}/propagators", propagator_nodes_check)

    
class job_contraction(g.jobs.base):
    def __init__(self, conf, conf_file, basis_dir, t, solver, dependencies):
        self.conf = conf
        self.t = t
        super().__init__(f"{conf}/pm_contr_{solver}_t{t}", dependencies)

    def perform(self, root):
        pass

    def check(self, root):
        return True

    
jobs = []

for group in groups:
    for conf in groups[group]["confs"]:
        conf_file = groups[group]["conf_fmt"] % conf
        evec_dir = groups[group]["evec_fmt"] % conf
        for t in [0,2]:#range(0, 192, 2):
            basis_dir = groups[group]["basis_fmt"] % (conf,t)

            # first need perambulators for each time-slice
            dep_group = []
            for i0 in range(0, basis_size, sloppy_per_job):
                j = job_perambulator(conf, evec_dir, conf_file, basis_dir, t, i0, "sloppy")
                jobs.append(j)
                dep_group.append(j.name)

            # then once time-slice is complete, perform contractions and compress perambulators
            jc = job_contraction(conf, conf_file, basis_dir, t, "sloppy", dep_group)
            jobs.append(jc)
            dep_group.append(jc.name)

            # once all of that is done, delete full perambulators (simple delete job)

        break
    break


# main job loop
for jid in range(g.default.get_int("--gpt_jobs", 1)):
    g.jobs.next(root_output, jobs)


