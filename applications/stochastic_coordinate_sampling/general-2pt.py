#!/usr/bin/env python3
#
# Content: general 2pt mesonic contractions from sparse point source data sets
# Author: Christoph Lehner
# Date: March 2026
#

import gpt as g
import numpy as np
import sys

conf = g.default.get_int("--conf", None)

cache_params = (512, 64)

output = g.corr_io.writer(f"general-2pt.{conf}")

for tag in ["ems","s"]:

    g.message(
        f"""

    Test {tag}

    """
    )
    weights, light = g.qcd.sparse_propagator.flavor(f"scratch/ensemble-Ca/propagators/{conf}_combined/light.{tag}", *cache_params)
    _, strange = g.qcd.sparse_propagator.flavor(f"scratch/ensemble-Ca/propagators/{conf}_combined/strange.{tag}", *cache_params)

    g.message(weights)

    coordinates = light.sink_domain.coordinates
    nsrc = light.source_domain.sampled_sites
    nsnk = light.sink_domain.sampled_sites - 1 # remove source from sink sampling to have it correct at t=0
    sdomain = light.sink_domain.sdomain
    V = light.sink_domain.total_sites

    Basis=[
        ("I", g.gamma["I"]),
        ("G5", g.gamma[5])
    ]
    for mu in range(4):
        Basis.append((f"G{mu}", g.gamma[mu]))
        Basis.append((f"G{mu}G5", g.gamma[mu].tensor()*g.gamma[5].tensor()))
        for nu in range(mu):
            Basis.append((f"G{mu}G{nu}", g.gamma[mu].tensor()*g.gamma[nu].tensor()))

    def process(cors, pos, q1, q2, tg, w):
        for tag_sink, Gam_sink in Basis:
            for tag_src, Gam_src in Basis:
                Gam_tag=f"sink_{tag_sink}_source_{tag_src}"
                cor=sdomain.slice(g.trace(Gam_sink * q1 * Gam_src * g.gamma[5] * g.adj(q2) * g.gamma[5]), 3)
                # TODO: add momenta?
                cor=cor[pos[3]:] + cor[:pos[3]]
                tg_full=f"{tg}/{pos}/{Gam_tag}/p_0.0.0"
                if tg_full not in cors:
                    cors[tg_full] = np.array(cor) * w * (V/nsnk)
                else:
                    cors[tg_full] += np.array(cor) * w * (V/nsnk)

    g.message(f"Sparsening factor: {V/nsnk}")
    
    for i in range(nsrc):

        pos_i = coordinates[i]

        g.message(f"Process {i} / {nsrc}: {pos_i}")

        correlators = {}
        t = g.timer()
        for w, [tag_prec] in weights:

            t("propagator")
            light_i = light[tag_prec, i, [i]]
            strange_i = strange[tag_prec, i, [i]]

            t("contract")
            process(correlators, pos_i, light_i, light_i, "light.light", w)
            process(correlators, pos_i, light_i, strange_i, "light.strange", w)
            process(correlators, pos_i, strange_i, light_i, "strange.light", w)
            process(correlators, pos_i, strange_i, strange_i, "strange.strange", w)
            t()

        for c in correlators:
            output.write(f"{tag}/{c}", correlators[c])

        g.message(t)
