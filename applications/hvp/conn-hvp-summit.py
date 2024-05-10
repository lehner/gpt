#!/usr/bin/env python3
#
# Authors: Christoph Lehner 2020
#
# Calculate HVP connected diagram with A2A method
#
import gpt as g
import sys, os

# configure
root_output = "/gpfs/alpine/phy157/proj-shared/phy157dwf/lehner/conn-hvp"

groups = {
    "phy131": {
        "confs": ["420", "500", "580"],
        "evec_fmt": "/gpfs/alpine/phy157/proj-shared/phy157dwf/lehner/evec-cache/96I/%s/lanczos.output",
        "conf_fmt": "/gpfs/alpine/phy138/proj-shared/phy138flavor/chulwoo/evols/96I2.8Gev/evol0/configurations/ckpoint_lat.%s",
    },
    "phy138": {
        "confs": ["460", "640", "540", "720"],
        "evec_fmt": "/gpfs/alpine/phy157/proj-shared/phy157dwf/lehner/evec-cache/96I/%s/lanczos.output",
        "conf_fmt": "/gpfs/alpine/phy138/proj-shared/phy138flavor/chulwoo/evols/96I2.8Gev/evol0/configurations/ckpoint_lat.%s",
    },
    "phy138n": {
        "confs": ["480", "520", "560", "600", "620"],
        "evec_fmt": "/gpfs/alpine/phy157/proj-shared/phy157dwf/lehner/evec-cache/96I/%s/lanczos.output",
        "conf_fmt": "/gpfs/alpine/phy138/proj-shared/phy138flavor/chulwoo/evols/96I2.8Gev/evol0/configurations/ckpoint_lat.%s",
    },
}

jobs = {
    "exact_0": {
        "exact": 1,
        "sloppy": 0,
        "low": 0,
        "all_time_slices": True,
    },  # 1270 seconds + 660 to load ev
    "sloppy_0": {
        "exact": 0,
        "sloppy": 8,
        "low": 0,
        "all_time_slices": True,
    },  # 2652 seconds + 580 to load ev
    "sloppy_1": {
        "exact": 0,
        "sloppy": 8,
        "low": 0,
        "all_time_slices": True,
    },  # 2652 seconds + 580 to load ev
    "low_0": {
        "exact": 0,
        "sloppy": 0,
        "low": 150,
        "all_time_slices": True,
    },  # 2100 seconds + 600 to load ev
    "low_1": {"exact": 0, "sloppy": 0, "low": 150, "all_time_slices": True},
    "exact_0_correlated": {
        "exact": 1,
        "sloppy": 0,
        "low": 0,
        "all_time_slices": False,
    },  # 1270 seconds + 660 to load ev
    "sloppy_0_correlated": {
        "exact": 0,
        "sloppy": 8,
        "low": 0,
        "all_time_slices": False,
    },  # 2652 seconds + 580 to load ev
    "low_0_correlated": {
        "exact": 0,
        "sloppy": 0,
        "low": 150,
        "all_time_slices": False,
    },  # 2100 seconds + 600 to load ev
    "sloppy_2": {
        "exact": 0,
        "sloppy": 8,
        "low": 0,
        "all_time_slices": True,
    },  # 2652 seconds + 580 to load ev
    "sloppy_3": {
        "exact": 0,
        "sloppy": 8,
        "low": 0,
        "all_time_slices": True,
    },  # 2652 seconds + 580 to load ev
    "low_2": {
        "exact": 0,
        "sloppy": 0,
        "low": 150,
        "all_time_slices": True,
    },  # 2100 seconds + 600 to load ev
    "low_3": {"exact": 0, "sloppy": 0, "low": 150, "all_time_slices": True},
    "sloppy_1_correlated": {
        "exact": 0,
        "sloppy": 8,
        "low": 0,
        "all_time_slices": False,
    },  # 2652 seconds + 580 to load ev
    "low_1_correlated": {
        "exact": 0,
        "sloppy": 0,
        "low": 150,
        "all_time_slices": False,
    },  # 2100 seconds + 600 to load ev
    "sloppy_4": {
        "exact": 0,
        "sloppy": 8,
        "low": 0,
        "all_time_slices": True,
    },  # 2652 seconds + 580 to load ev
    "sloppy_5": {
        "exact": 0,
        "sloppy": 8,
        "low": 0,
        "all_time_slices": True,
    },  # 2652 seconds + 580 to load ev
    "sloppy_6": {
        "exact": 0,
        "sloppy": 8,
        "low": 0,
        "all_time_slices": True,
    },  # 2652 seconds + 580 to load ev
    "sloppy_7": {
        "exact": 0,
        "sloppy": 8,
        "low": 0,
        "all_time_slices": True,
    },  # 2652 seconds + 580 to load ev
    "low_4": {
        "exact": 0,
        "sloppy": 0,
        "low": 150,
        "all_time_slices": True,
    },  # 2100 seconds + 600 to load ev
    "low_5": {"exact": 0, "sloppy": 0, "low": 150, "all_time_slices": True},
    "low_6": {
        "exact": 0,
        "sloppy": 0,
        "low": 150,
        "all_time_slices": True,
    },  # 2100 seconds + 600 to load ev
    "low_7": {"exact": 0, "sloppy": 0, "low": 150, "all_time_slices": True},
    "sloppy_2_correlated": {
        "exact": 0,
        "sloppy": 8,
        "low": 0,
        "all_time_slices": False,
    },  # 2652 seconds + 580 to load ev
    "low_2_correlated": {
        "exact": 0,
        "sloppy": 0,
        "low": 150,
        "all_time_slices": False,
    },  # 2100 seconds + 600 to load ev
    "sloppy_3_correlated": {
        "exact": 0,
        "sloppy": 8,
        "low": 0,
        "all_time_slices": False,
    },  # 2652 seconds + 580 to load ev
    "low_3_correlated": {
        "exact": 0,
        "sloppy": 0,
        "low": 150,
        "all_time_slices": False,
    },  # 2100 seconds + 600 to load ev
    "sloppy_8": {
        "exact": 0,
        "sloppy": 8,
        "low": 0,
        "all_time_slices": True,
    },  # 2652 seconds + 580 to load ev
    "sloppy_9": {
        "exact": 0,
        "sloppy": 8,
        "low": 0,
        "all_time_slices": True,
    },  # 2652 seconds + 580 to load ev
    "sloppy_10": {
        "exact": 0,
        "sloppy": 8,
        "low": 0,
        "all_time_slices": True,
    },  # 2652 seconds + 580 to load ev
    "sloppy_11": {
        "exact": 0,
        "sloppy": 8,
        "low": 0,
        "all_time_slices": True,
    },  # 2652 seconds + 580 to load ev
    "low_8": {
        "exact": 0,
        "sloppy": 0,
        "low": 150,
        "all_time_slices": True,
    },  # 2100 seconds + 600 to load ev
    "low_9": {"exact": 0, "sloppy": 0, "low": 150, "all_time_slices": True},
    "low_10": {
        "exact": 0,
        "sloppy": 0,
        "low": 150,
        "all_time_slices": True,
    },  # 2100 seconds + 600 to load ev
    "low_11": {"exact": 0, "sloppy": 0, "low": 150, "all_time_slices": True},
    "sloppy_4_correlated": {
        "exact": 0,
        "sloppy": 8,
        "low": 0,
        "all_time_slices": False,
    },  # 2652 seconds + 580 to load ev
    "low_4_correlated": {
        "exact": 0,
        "sloppy": 0,
        "low": 150,
        "all_time_slices": False,
    },  # 2100 seconds + 600 to load ev
    "sloppy_5_correlated": {
        "exact": 0,
        "sloppy": 8,
        "low": 0,
        "all_time_slices": False,
    },  # 2652 seconds + 580 to load ev
    "low_5_correlated": {
        "exact": 0,
        "sloppy": 0,
        "low": 150,
        "all_time_slices": False,
    },  # 2100 seconds + 600 to load ev
}

# At 32 jobs we break even with eigenvector generation

simultaneous_low_positions = 3

jobs_per_run = g.default.get_int("--gpt_jobs", 1)

source_time_slices = 2

save_propagators = True

operators = {
    "G0": g.gamma[0],
    "G1": g.gamma[1],
    "G2": g.gamma[2],
    "G5": g.gamma[5],
}

correlators = [
    ("G0", "G0"),
    ("G1", "G1"),
    ("G2", "G2"),
    ("G5", "G5"),
]


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

    return None


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
       HVP connected run on summit ;  this run will attempt:
================================================================================
"""
)
for group, job, conf, jid, n in run_jobs:
    g.message(
        f"""

    Job {jid} / {n} :  configuration {conf}, job tag {job}

"""
    )


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

light_innerL_inverter = g.algorithms.inverter.preconditioned(
    g.qcd.fermion.preconditioner.eo1_ne(parity=g.odd),
    g.algorithms.inverter.sequence(
        g.algorithms.inverter.coarse_deflate(
            eig[1],
            eig[0],
            eig[2],
            block=200,
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
            block=200,
        ),
        g.algorithms.inverter.split(
            g.algorithms.inverter.cg({"eps": 1e-8, "maxiter": 300}),
            mpi_split=g.default.get_ivec("--mpi_split", None, 4),
        ),
    ),
)

light_low_inverter = g.algorithms.inverter.preconditioned(
    g.qcd.fermion.preconditioner.eo1_ne(parity=g.odd),
    g.algorithms.inverter.coarse_deflate(eig[1], eig[0], eig[2], block=200),
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
prop_l_sloppy = l_exact.propagator(light_sloppy_inverter)
prop_l_exact = l_exact.propagator(light_exact_inverter)

# show available memory
g.mem_report(details=False)

# per job
for group, job, conf, jid, n in run_jobs:
    g.message(
        f"""

    Job {jid} / {n} :  configuration {conf}, job tag {job}

"""
    )

    job_seed = job.split("_correlated")[0]
    rng = g.random(f"hvp-conn-a2a-ensemble-{conf}-{job_seed}")

    source_positions_low = [
        [rng.uniform_int(min=0, max=L[i] - 1) for i in range(4)] for j in range(jobs[job]["low"])
    ]
    source_positions_sloppy = [
        [rng.uniform_int(min=0, max=L[i] - 1) for i in range(4)] for j in range(jobs[job]["sloppy"])
    ]
    source_positions_exact = [
        [rng.uniform_int(min=0, max=L[i] - 1) for i in range(4)] for j in range(jobs[job]["exact"])
    ]

    all_time_slices = jobs[job]["all_time_slices"]
    use_source_time_slices = source_time_slices
    if not all_time_slices:
        use_source_time_slices = 1

    g.message(f" positions_low = {source_positions_low}")
    g.message(f" positions_sloppy = {source_positions_sloppy}")
    g.message(f" positions_exact = {source_positions_exact}")
    g.message(f" all_time_slices = {all_time_slices}")

    root_job = f"{root_output}/{conf}/{job}"
    output = g.gpt_io.writer(f"{root_job}/propagators")
    output_correlator = g.corr_io.writer(f"{root_job}/head.dat")

    # contractor
    def contract(pos, prop, tag, may_save_prop=True):
        t0 = pos[3]
        prop_tag = "%s/%s" % (tag, str(pos))
        if not all_time_slices:
            prop_tag = "single_time_slice/" + prop_tag

        # save propagators
        if save_propagators and may_save_prop:
            output.write({prop_tag: prop})
            output.flush()

        # create and save correlators
        for op_snk, op_src in correlators:
            G_snk = operators[op_snk]
            G_src = operators[op_src]
            corr = g.slice(g.trace(G_src * g.gamma[5] * g.adj(prop) * g.gamma[5] * G_snk * prop), 3)
            corr = corr[t0:] + corr[:t0]

            corr_tag = "%s/snk%s-src%s" % (prop_tag, op_snk, op_src)
            output_correlator.write(corr_tag, corr)
            g.message("Correlator %s\n" % corr_tag, corr)

    # prepare sources
    vol3d = (
        l_exact.U_grid.fdimensions[0]
        * l_exact.U_grid.fdimensions[1]
        * l_exact.U_grid.fdimensions[2]
    )

    full_time = l_exact.U_grid.fdimensions[3]
    assert full_time % source_time_slices == 0
    sparse_time = full_time // source_time_slices

    # source creation
    def create_source(pos, point=False):
        srcD = g.mspincolor(l_exact.U_grid)
        srcD[:] = 0

        # create time-sparsened source
        sign_of_slice = [rng.zn(n=2) for i in range(source_time_slices)]
        for i in range(use_source_time_slices, source_time_slices):
            sign_of_slice[i] = 0.0

        pos_of_slice = [
            [pos[i] if i < 3 else (pos[i] + j * sparse_time) % full_time for i in range(4)]
            for j in range(source_time_slices)
        ]
        g.message(f"Signature: {pos} -> {pos_of_slice} with signs {sign_of_slice}")
        for i in range(source_time_slices):
            if point:
                srcD += g.create.point(g.lattice(srcD), pos_of_slice[i]) * sign_of_slice[i]
            else:
                srcD += g.create.wall.z2(g.lattice(srcD), pos_of_slice[i][3], rng) * (
                    sign_of_slice[i] / vol3d**0.5
                )

        return srcD, pos_of_slice, sign_of_slice

    # exact positions
    for pos in source_positions_exact:
        srcD, pos_of_slice, sign_of_slice = create_source(pos)
        srcF = g.convert(srcD, g.single)

        prop_exact = g.eval(prop_l_exact * srcD)
        prop_sloppy = g.eval(prop_l_sloppy * srcD)
        prop_low = g.eval(prop_l_low * srcF)

        for i in range(use_source_time_slices):
            contract(pos_of_slice[i], g.eval(sign_of_slice[i] * prop_exact), "exact")
            contract(pos_of_slice[i], g.eval(sign_of_slice[i] * prop_sloppy), "sloppy")
            contract(pos_of_slice[i], g.eval(sign_of_slice[i] * prop_low), "low", False)

    # sloppy positions
    for pos in source_positions_sloppy:
        srcD, pos_of_slice, sign_of_slice = create_source(pos)
        srcF = g.convert(srcD, g.single)

        prop_sloppy = g.eval(prop_l_sloppy * srcD)
        prop_low = g.eval(prop_l_low * srcF)

        for i in range(use_source_time_slices):
            contract(pos_of_slice[i], g.eval(sign_of_slice[i] * prop_sloppy), "sloppy")
            contract(pos_of_slice[i], g.eval(sign_of_slice[i] * prop_low), "low", False)

    # low positions
    for pos_idx in range(0, len(source_positions_low), simultaneous_low_positions):
        array_srcF = []
        array_pos_of_slice = []
        array_sign_of_slice = []

        for inner in range(simultaneous_low_positions):
            srcD, pos_of_slice, sign_of_slice = create_source(
                source_positions_low[pos_idx + inner], point=True
            )
            srcF = g.convert(srcD, g.single)
            array_srcF.append(srcF)
            array_pos_of_slice.append(pos_of_slice)
            array_sign_of_slice.append(sign_of_slice)

        array_prop_low = g.eval(prop_l_low * array_srcF)

        for inner in range(simultaneous_low_positions):
            prop_low = array_prop_low[inner]
            pos_of_slice = array_pos_of_slice[inner]
            sign_of_slice = array_sign_of_slice[inner]
            for i in range(use_source_time_slices):
                contract(
                    pos_of_slice[i],
                    g.eval(sign_of_slice[i] * prop_low),
                    "low-pnt",
                    False,
                )
