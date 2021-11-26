#!/usr/bin/env python3
#
# Authors: Christoph Lehner 2020
#
# Calculate HVP connected diagram with A2A method
#
import gpt as g
import numpy as np
import sys, os, glob
import shutil

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

T = 192
propagator_nodes_check = 1024
compressed_propagator_nodes_check = 64
sloppy_per_job = 50
basis_size = 200
compress_ratio = 0.01

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
    
        self.prop_l_low = config.l_sloppy.propagator(light_low_inverter)
        self.prop_l_sloppy = config.l_exact.propagator(light_sloppy_inverter).grouped(2)
        self.prop_l_exact = config.l_exact.propagator(light_exact_inverter).grouped(2)


def propagator_consistency_check(dn, n0):
    fs = glob.glob(f"{dn}/??/*.field")
    n = len(fs)
    g.message("Check",n0,n)
    if n != n0:
        return False
    s0 = os.path.getsize(fs[0])
    if s0 == 0:
        return False
    szOK = all([ s0 == os.path.getsize(f) for f in fs[1:] ])
    g.message("Check",szOK,s0)
    return szOK


class job_perambulator(g.jobs.base):
    def __init__(self, conf, evec_dir, conf_file, basis_dir, t, i0, solver, dependencies):
        self.conf = conf
        self.solver = solver
        self.conf_file = conf_file
        self.evec_dir = evec_dir
        self.basis_dir = basis_dir
        self.t = t
        self.i0 = i0
        self.ilist = list(range(i0, i0+sloppy_per_job))
        super().__init__(f"{conf}/pm_{solver}_t{t}_i{i0}", dependencies)
        self.weight = 1.0

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
        
        vcj = g.load(f"{root}/{self.conf}/pm_basis/basis")
        c = g.coordinates(vcj[0])
        c = c[c[:,3] == self.t]

        g.message(f"t = {self.t}, ilist = {self.ilist}, basis size = {len(vcj)}, solver = {self.solver}")

        root_job = f"{root}/{self.name}"
        output = g.gpt_io.writer(f"{root_job}/propagators")
    
        # create sources
        srcD = [g.vspincolor(current_config.l_exact.U_grid) for spin in range(4)]

        for i in self.ilist:
        
            for spin in range(4):
                srcD[spin][:] = 0
                srcD[spin][c,spin,:] = vcj[i][c]

                g.message("Norm of source:",g.norm2(srcD[spin]))
                if i == 0:
                    g.message("Source at origin:", srcD[spin][0,0,0,0])
                    g.message("Source at time-origin:", srcD[spin][0,0,0,self.t])
            
            prop = g.eval(prop_l * srcD)
            g.mem_report(details=False)

            for spin in range(4):
                output.write({f"t{self.t}s{spin}c{i}_{self.solver}": prop[spin]})
                output.flush()

    def check(self, root):
        return propagator_consistency_check(f"{root}/{self.name}/propagators", propagator_nodes_check)


class job_basis_layout(g.jobs.base):
    def __init__(self, conf, conf_file, basis_fmt, dependencies):
        self.conf = conf
        self.conf_file = conf_file
        self.basis_fmt = basis_fmt
        super().__init__(f"{conf}/pm_basis", dependencies)

    def perform(self, root):
        global basis_size, T, current_config
        if current_config is not None and current_config.conf_file != self.conf_file:
            current_config = None
        if current_config is None:
            current_config = config(self.conf_file)

        c = None
        vcj = [g.vcolor(current_config.l_exact.U_grid) for jr in range(basis_size)]
        for vcjj in vcj:
            vcjj[:] = 0

        for tprime in range(T):
            basis_evec, basis_evals = g.load(self.basis_fmt % (self.conf,tprime))
                    
            cache = {}
            plan = g.copy_plan(vcj[0], basis_evec[0], embed_in_communicator=vcj[0].grid)
            c = g.coordinates(basis_evec[0])
            plan.destination += vcj[0].view[np.hstack((c, np.ones((len(c),1),dtype=np.int32)*tprime))]
            plan.source += basis_evec[0].view[c]
            plan = plan()
                    
            for l in range(basis_size):
                plan(vcj[l], basis_evec[l])

        for l in range(basis_size):
            g.message("Check norm:",l,g.norm2(vcj[l]))
            
        g.save(f"{root}/{self.name}/basis", vcj)
        
    def check(self, root):
        return propagator_consistency_check(f"{root}/{self.name}/basis", propagator_nodes_check)


class job_contraction(g.jobs.base):
    def __init__(self, conf, conf_file, t, solver, dependencies):
        self.conf = conf
        self.conf_file = conf_file
        self.t = t
        self.solver = solver
        super().__init__(f"{conf}/pm_contr_{solver}_t{t}", dependencies)
        self.weight = 0.5

    def perform(self, root):
        global basis_size, sloppy_per_job, T, current_config
        if current_config is not None and current_config.conf_file != self.conf_file:
            current_config = None
        if current_config is None:
            current_config = config(self.conf_file)

        output_correlator = g.corr_io.writer(f"{root}/{self.name}/head.dat")

        vcj = g.load(f"{root}/{self.conf}/pm_basis/basis")
                
        for i0 in range(0, basis_size, sloppy_per_job):
            half_peramb = {}
            for l in g.load(f"{root}/{self.conf}/pm_{self.solver}_t{self.t}_i{i0}/propagators"):
                for x in l:
                    half_peramb[x] = l[x]

            g.mem_report(details=False)

            vc = g.vcolor(vcj[0].grid)
            c = g.coordinates(vc)
            prec = {"sloppy":0,"exact":1}[self.solver]
            
            for spin_prime in range(4):

                plan = None

                for spin in range(4):

                    for i in range(i0, i0 + sloppy_per_job):
                        hp = half_peramb[f"t{self.t}s{spin}c{i}_{self.solver}"]

                        if plan is None:
                            plan = g.copy_plan(vc, hp)
                            plan.destination += vc.view[c]
                            plan.source += hp.view[c,spin_prime,:]
                            plan = plan()

                        plan(vc, hp)

                        t0 = g.time()
                        slc_j = [g(g.adj(vcj[j]) * vc) for j in range(basis_size)]
                        t1 = g.time()
                        slc = g.slice(slc_j, 3)
                        t2 = g.time()

                        for j in range(basis_size):
                            output_correlator.write(f"output/peramb_prec{prec}/n_{j}_{i}_s_{spin_prime}_{spin}_t_{self.t}", slc[j])

                        t3 = g.time()
                        if i % 50 == 0:
                            g.message(spin_prime,spin,i,"Timing",t1-t0,t2-t1,t3-t2)

        output_correlator.close()
        
    def check(self, root):
        cnt = g.corr_io.count(f"{root}/{self.name}/head.dat")
        exp = basis_size ** 2 * 16
        g.message("check count",cnt,exp)
        return cnt == exp


class job_mom(g.jobs.base):
    def __init__(self, conf, conf_file, mom, tag, dependencies):
        self.conf = conf
        self.conf_file = conf_file
        self.mom = mom
        super().__init__(f"{conf}/pm_mom_{tag}", dependencies)

    def perform(self, root):
        global basis_size, T

        output_correlator = g.corr_io.writer(f"{root}/{self.name}/head.dat")
        
        vcj = g.load(f"{root}/{self.conf}/pm_basis/basis")

        for m in self.mom:
            mom_str = "_".join([str(x) for x in m])
            p = np.array([2.*np.pi/vcj[0].grid.gdimensions[i] * m[i] for i in range(3)] + [0])

            phase = g.complex(vcj[0].grid)
            phase[:] = 1
            phase @= g.exp_ixp(p) * phase

            g.message("L = ",vcj[0].grid.gdimensions)
            g.message("Momentum",p,m)

            for n in range(basis_size):
                t0 = g.time()
                vc_n = g(phase * vcj[n])
                t1 = g.time()
                slc_nprime = [g(g.adj(vcj[nprime]) * vc_n) for nprime in range(basis_size)]
                t2 = g.time()
                slc = g.slice(slc_nprime, 3)
                t3 = g.time()
            
                for nprime in range(basis_size):
                    output_correlator.write(f"output/mom/{mom_str}_n_{nprime}_{n}", slc[nprime])

                t4 = g.time()
            
                if n % 10 == 0:
                    g.message(n,"Timing",t1-t0,t2-t1,t3-t2,t4-t3)

        output_correlator.close()
        
    def check(self, root):
        cnt = g.corr_io.count(f"{root}/{self.name}/head.dat")
        exp = basis_size ** 2 * len(self.mom)
        g.message("check count",cnt,exp)
        return cnt == exp

class job_local_insertion(g.jobs.base):
    def __init__(self, conf, conf_file, t, solver, dependencies):
        self.conf = conf
        self.conf_file = conf_file
        self.t = t
        self.solver = solver
        super().__init__(f"{conf}/pm_local_insertion_{solver}_t{t}", dependencies)
        self.weight = 1.5
        
    def perform(self, root):
        global basis_size, sloppy_per_job, T, current_config
        if current_config is not None and current_config.conf_file != self.conf_file:
            current_config = None
        if current_config is None:
            current_config = config(self.conf_file)

        output_correlator = g.corr_io.writer(f"{root}/{self.name}/head.dat")

        # <np,sp| D^{-1} Gamma D^{-1} |n,s> = < (D^{-1})^\dagger |np,sp> | Gamma | D^{-1} |n,s > >
        # = < Gamma5 D^{-1} Gamma5 |np,sp> | Gamma | D^{-1} |n,s > >
        # = < D^{-1} |np,sp> | Gamma5 Gamma | D^{-1} |n,s > > gamma5_sign[sp]
        gamma5_sign = [1.,1.,-1.,-1.]
        indices = [0,1,2,5]
        prec = {"sloppy":0,"exact":1}[self.solver]
        
        for i0 in range(0, basis_size, sloppy_per_job):
            half_peramb_i = {}
            for l in g.load(f"{root}/{self.conf}/pm_{self.solver}_t{self.t}_i{i0}/propagators"):
                for x in l:
                    half_peramb_i[x] = l[x]
                    
            for j0 in range(0, basis_size, sloppy_per_job):
                if j0 == i0:
                    half_peramb_j = half_peramb_i
                else:
                    half_peramb_j = {}
                    for l in g.load(f"{root}/{self.conf}/pm_{self.solver}_t{self.t}_i{j0}/propagators"):
                        for x in l:
                            half_peramb_j[x] = l[x]

                for i in range(i0, i0 + sloppy_per_job):
                    for spin in range(4):
                        g.message(i, spin)
                        hp_i = half_peramb_i[f"t{self.t}s{spin}c{i}_{self.solver}"]
                        for mu in indices:
                            hp_i_gamma = g( g.gamma[5] * g.gamma[mu] * hp_i )
                            for spin_prime in range(4):
                                slc_j = [g(gamma5_sign[spin_prime]*g.adj(half_peramb_j[f"t{self.t}s{spin_prime}c{j}_{self.solver}"])*hp_i_gamma)
                                         for j in range(j0, j0 + sloppy_per_job)]
                                slc = g.slice(slc_j, 3)
                               
                                for j in range(j0, j0 + sloppy_per_job):
                                    output_correlator.write(f"output/G{mu}_prec{prec}/n_{j}_{i}_s_{spin_prime}_{spin}_t_{self.t}", slc[j-j0])
                                    
        output_correlator.close()
        
    def check(self, root):
        cnt = g.corr_io.count(f"{root}/{self.name}/head.dat")
        exp = basis_size ** 2 * 16 * 4
        g.message("check count",cnt,exp)
        return cnt == exp


class job_compress_half_peramb(g.jobs.base):
    def __init__(self, conf, conf_file, t, solver, dependencies):
        self.conf = conf
        self.conf_file = conf_file
        self.t = t
        self.solver = solver
        super().__init__(f"{conf}/pm_compressed_half_peramb_{solver}_t{t}", dependencies)
        self.weight = 0.2

    def perform(self, root):
        global basis_size, sloppy_per_job, T, current_config, compress_ratio
        if current_config is not None and current_config.conf_file != self.conf_file:
            current_config = None
        if current_config is None:
            current_config = config(self.conf_file)

        U = current_config.U
        reduced_mpi = [x for x in U[0].grid.mpi]
        for i in range(len(reduced_mpi)):
            if reduced_mpi[i] % 2 == 0:
                reduced_mpi[i] //= 2

        # create random selection of points with same spatial sites on each sink time slice
        # use different spatial sites for each source time-slice
        # this should be optimal for the local operator insertions
        rng = g.random(f"sparse2_{self.conf}_{self.t}")
        grid = U[0].grid
        t0 = grid.ldimensions[3] * grid.processor_coor[3]
        t1 = t0 + grid.ldimensions[3]
        spatial_sites = int(compress_ratio * np.prod(grid.ldimensions[0:3]))
        spatial_coordinates = rng.choice(g.coordinates(U[0]), spatial_sites)
        local_coordinates = np.repeat(spatial_coordinates, t1-t0, axis=0)
        for t in range(t0, t1):
            local_coordinates[t-t0::t1-t0,3] = t
    
        sdomain = g.domain.sparse(
            current_config.l_exact.U_grid,
            local_coordinates
        )

        half_peramb = { "sparse_domain" : sdomain }
        for i0 in range(0, basis_size, sloppy_per_job):

            for l in g.load(f"{root}/{self.conf}/pm_{self.solver}_t{self.t}_i{i0}/propagators"):
                for x in l:

                    S = sdomain.lattice(l[x].otype)
                    sdomain.project(S, l[x])

                    half_peramb[x] = S

                    g.message(x)

        g.save(f"{root}/{self.name}/propagators", half_peramb, g.format.gpt(
            { "mpi": reduced_mpi }
            ))
        
    def check(self, root):
        return propagator_consistency_check(f"{root}/{self.name}/propagators", compressed_propagator_nodes_check)


class job_local_insertion_using_compressed(g.jobs.base):
    def __init__(self, conf, conf_file, t, solver, dependencies):
        self.conf = conf
        self.conf_file = conf_file
        self.t = t
        self.solver = solver
        super().__init__(f"{conf}/pm_local_insertion_using_compressed_{solver}_t{t}", dependencies)
        self.weight = 1.5
        
    def perform(self, root):
        global basis_size, sloppy_per_job, T, current_config
        if current_config is not None and current_config.conf_file != self.conf_file:
            current_config = None
        if current_config is None:
            current_config = config(self.conf_file)

        output_correlator = g.corr_io.writer(f"{root}/{self.name}/head.dat")

        # <np,sp| D^{-1} Gamma D^{-1} |n,s> = < (D^{-1})^\dagger |np,sp> | Gamma | D^{-1} |n,s > >
        # = < Gamma5 D^{-1} Gamma5 |np,sp> | Gamma | D^{-1} |n,s > >
        # = < D^{-1} |np,sp> | Gamma5 Gamma | D^{-1} |n,s > > gamma5_sign[sp]
        indices = [0,1,2,5]
        prec = {"sloppy":0,"exact":1}[self.solver]
        
        half_peramb = g.load(f"{root}/{self.conf}/pm_compressed_half_peramb_{self.solver}_t{self.t}/propagators")

        sdomain = half_peramb["sparse_domain"]
        scale = sdomain.grid.gsites / sdomain.grid.Nprocessors / len(sdomain.local_coordinates)
        g.message("scale =",scale)
        gamma5_sign = [1.*scale,1.*scale,-1.*scale,-1.*scale]

        for i0 in range(0, basis_size, sloppy_per_job):
            half_peramb_i = {}
            for i in range(i0, i0 + sloppy_per_job):
                for spin in range(4):
                    f = g.vspincolor(sdomain.grid)
                    f[:] = 0
                    sdomain.promote(f, half_peramb[f"t{self.t}s{spin}c{i}_{self.solver}"])
                    half_peramb_i[f"t{self.t}s{spin}c{i}_{self.solver}"] = f
                    
            for j0 in range(0, basis_size, sloppy_per_job):
                if j0 == i0:
                    half_peramb_j = half_peramb_i
                else:
                    half_peramb_j = {}
                    for j in range(j0, j0 + sloppy_per_job):
                        for spin in range(4):
                            f = g.vspincolor(sdomain.grid)
                            f[:] = 0
                            sdomain.promote(f, half_peramb[f"t{self.t}s{spin}c{j}_{self.solver}"])
                            half_peramb_j[f"t{self.t}s{spin}c{j}_{self.solver}"] = f

                for i in range(i0, i0 + sloppy_per_job):
                    for spin in range(4):
                        g.message(i, spin)
                        hp_i = half_peramb_i[f"t{self.t}s{spin}c{i}_{self.solver}"]
                        for mu in indices:
                            hp_i_gamma = g( g.gamma[5] * g.gamma[mu] * hp_i )
                            for spin_prime in range(4):
                                slc_j = [g(gamma5_sign[spin_prime]*g.adj(half_peramb_j[f"t{self.t}s{spin_prime}c{j}_{self.solver}"])*hp_i_gamma)
                                         for j in range(j0, j0 + sloppy_per_job)]
                                slc = g.slice(slc_j, 3)
                               
                                for j in range(j0, j0 + sloppy_per_job):
                                    output_correlator.write(f"output/G{mu}_prec{prec}/n_{j}_{i}_s_{spin_prime}_{spin}_t_{self.t}", slc[j-j0])
                                    
        output_correlator.close()
        
    def check(self, root):
        cnt = g.corr_io.count(f"{root}/{self.name}/head.dat")
        exp = basis_size ** 2 * 16 * 4
        g.message("check count",cnt,exp)
        return cnt == exp


class job_delete_half_peramb(g.jobs.base):
    def __init__(self, conf, t, targets, dependencies):
        self.targets = targets
        super().__init__(f"{conf}/pm_delete_t{t}",dependencies)
        self.weight = 0.1

    def perform(self, root):
        for target in self.targets:
            dst = f"{root}/{target}/propagators"
            g.message(dst,"Delete", os.path.exists(dst))
            if os.path.exists(dst) and g.rank() == 0:
                shutil.rmtree(dst)

        g.barrier()
        
    def check(self, root):
        for target in self.targets:
            dst = f"{root}/{target}/propagators"
            if os.path.exists(dst):
                return False

        return True


jobs = []

for group in groups:
    for conf in groups[group]["confs"]:
        conf_file = groups[group]["conf_fmt"] % conf
        evec_dir = groups[group]["evec_fmt"] % conf

        # first change basis layout
        jb = job_basis_layout(conf, conf_file, groups[group]["basis_fmt"], [])
        jobs.append(jb)

        # momentum projection
        momenta = []
        momenta_max_nsqr = 8
        for nx in range(-5,6):
            for	ny in range(-5,6):
                for nz in range(-5,6):
                    nsqr = nx**2 + ny**2 + nz**2
                    if nsqr <= momenta_max_nsqr:
                        momenta.append([nx,ny,nz])
        g.message(len(momenta))
        g.message(momenta)
        jm = job_mom(conf, conf_file, momenta, "first", [jb.name])
        jobs.append(jm)

        for t in range(0, 96, 2):
            basis_dir = groups[group]["basis_fmt"] % (conf,t)
       
            # first need perambulators for each time-slice
            dep_group = [jb.name]
            delete_names = []
            for i0 in range(0, basis_size, sloppy_per_job):
                j = job_perambulator(conf, evec_dir, conf_file, basis_dir, t, i0, "sloppy", [jb.name])
                jobs.append(j)
                dep_group.append(j.name)
                delete_names.append(j.name)
 
            # contract perambulators
            jc = job_contraction(conf, conf_file, t, "sloppy", dep_group)
            jobs.append(jc)

            # compress half-perambulators
            jcmp = job_compress_half_peramb(conf, conf_file, t, "sloppy", dep_group)
            jobs.append(jcmp)

            # contract local operator insertions
            jl = job_local_insertion(conf, conf_file, t, "sloppy", dep_group)
            jobs.append(jl)

            # contract local operator insertions using compressed
            # jlc = job_local_insertion_using_compressed(conf, conf_file, t, "sloppy", dep_group + [jcmp.name])
            # jobs.append(jlc)

            # once all of that is done, delete full perambulators (simple delete job)
            # dep_all = dep_group + [jc.name, jcmp.name, jl.name, jlc.name]
            dep_all = dep_group + [jc.name, jcmp.name, jl.name]
            j = job_delete_half_peramb(conf, t, delete_names, dep_all)
            jobs.append(j)
            


# main job loop
jobs_total = g.default.get_int("--gpt_jobs", 1) * 3
jobs_acc = 0
while jobs_acc < jobs_total:
    g.message("Weight left:", jobs_total - jobs_acc)
    j = g.jobs.next(root_output, jobs, max_weight = jobs_total - jobs_acc, stale_seconds = 3600 * 7)
    if j is None:
        break
    jobs_acc += j.weight
