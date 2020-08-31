#!/usr/bin/env python3
#
# Authors: Christoph Lehner 2020
#
# Desc.: Quantum Circuits
#
import gpt as g
import numpy as np
import sys, cgpt


def qubit_state(register_size):
    return g.complex(g.grid([2 ** n for n in register_size], g.double))


# 3,1 state
state = qubit_state([3, 1, 1])

# could split in mpi over first register

# initialize to |000,0,0>
state[:] = 0
state[0, 0, 0] = 1

# apply gates
# X(0) * state

g.message(state)
