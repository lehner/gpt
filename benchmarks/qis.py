#!/usr/bin/env python3
#
# Authors: Christoph Lehner 2020
#
# Desc.: Test basic QIS interface
#
import gpt as g
import numpy as np
from gpt.qis.gate import *

# need a random number generator for measurements
r = g.random("qis_test", "vectorized_ranlux24_24_64")


n = g.default.get_int("--n", 16)
N = g.default.get_int("--N", 10)

for precision in [g.single, g.double]:
    g.mem_report()
    for q in [g.qis.backends.dynamic]:  # g.qis.backends.static,
        g.message(
            f"""

    Run tests with

     {n} qubits
     {precision.__name__} precision
     {q.__name__} backend

            """
        )
        stR = q.state(r, n, precision=precision)
        stR.randomize()

        # test performance of one-qubit gates
        def Z(i):
            return R_z(i, np.pi)

        for gate in [X, H, Z]:
            for bit in range(n):
                g.message(
                    f"""
    - Benchmark {gate.__name__}({bit}) gate"""
                )

                circuit = gate(bit)
                for j in range(1, N):
                    circuit = circuit | gate(bit)
                t0 = g.time()
                st = circuit * stR
                t1 = g.time()
                GB = st.lattice.global_bytes() * N * 2 / 1e9
                g.message(
                    f"""
                    Number of applications :   {N}
                    Timing                 :   {t1-t0} s
                    Effective bandwidth    :   {GB/(t1-t0)} GB/s"""
                )

        # test performance of two-qubit gates
        for gate in [CNOT]:
            g.message(
                f"""
                - Benchmark {gate.__name__} gate"""
            )
            circuit = gate(0, 1)
            for j in range(1, N):
                idx = j % (n * n)
                control = idx // n
                target = idx % n
                if control == target:
                    target = (control + 1) % n
                circuit = circuit | gate(control, target)
            t0 = g.time()
            st = circuit * stR
            t1 = g.time()
            GB = st.lattice.global_bytes() * N * 2 / 1e9
            g.message(
                f"""
    Number of applications :   {N}
    Timing                 :   {t1-t0} s
    Effective bandwidth    :   {GB/(t1-t0)} GB/s"""
            )
