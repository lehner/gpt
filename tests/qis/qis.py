#!/usr/bin/env python3
#
# Authors: Christoph Lehner 2020
#
# Desc.: Test basic QIS interface
#
import gpt as g
import numpy as np
from gpt.qis.state import state
from gpt.qis.gate import *

# need a random number generator for measurements
r = g.random("qis_test")

# tests done for N qubits
N = 11
g.message(f"Run tests with {N} qubits")
st0 = state(r, N, precision=g.double)
stR = st0.cloned()
stR.randomize()


def check_same(state_a, state_b):
    assert (
        g.norm2(state_a.lattice - state_b.lattice) ** 0.5
        < state_a.lattice.grid.precision.eps
    )


def check_norm(state):
    assert (g.norm2(state.lattice) - 1.0) < state.lattice.grid.precision.eps


def set_value_of_qubit(state, i, val):
    s = state.cloned()
    mv = s.measure(i)
    if mv != val:
        s.X(i)
    return s


# first make sure state is properly initialized
g.message("Test initial state")
psi = M() * st0
assert psi.classical_bit == [0] * N

# now test all X gates
g.message("Test X")
for i in range(N):
    psi = (X(i) | M()) * st0
    ref = [0] * N
    ref[i] = 1
    assert psi.classical_bit == ref

# now test X^2 = 1
g.message("Test X^2")
for i in range(N):
    psi = (X(i) | X(i)) * stR
    check_same(psi, stR)

# now test H^2 = 1
g.message("Test H^2")
for i in range(N):
    psi = (H(i) | H(i)) * stR
    check_same(psi, stR)

# now test CNOT^2 = 1
g.message("Test CNOT^2")
for i in range(N):
    for j in range(N):
        if i != j:
            psi = (CNOT(i, j) | CNOT(i, j)) * stR
            check_same(psi, stR)

# now test S^4 = 1
g.message("Test S^4")
for i in range(N):
    psi = (
        R_z(i, np.pi / 2.0)
        | R_z(i, np.pi / 2.0)
        | R_z(i, np.pi / 2.0)
        | R_z(i, np.pi / 2.0)
    ) * stR
    check_same(psi, stR)

# check norm of all gates
g.message("Test gate unitarity")
for i in range(N):
    check_norm(X(i) * stR)
    check_norm(R_z(i, 0.125124) * stR)
    check_norm(H(i) * stR)
    for j in range(N):
        if i != j:
            check_norm(CNOT(i, j) * stR)

# H | Z | H = X
g.message("Test H | Z | H == X")
for i in range(N):
    a = (H(i) | R_z(i, np.pi) | H(i)) * stR
    b = X(i) * stR
    check_same(a, b)

# CNOT
g.message("Randomized CNOT test")
for control in range(N):
    # on a random vector with fixed control qubit value, perform the test
    state_control_0 = set_value_of_qubit(stR, control, 0)
    state_control_1 = set_value_of_qubit(stR, control, 1)
    for target in range(N):
        if target != control:
            psi = CNOT(control, target) * state_control_0
            check_same(psi, state_control_0)

            psi = (CNOT(control, target) | X(target)) * state_control_1
            check_same(psi, state_control_1)

# Bell state
g.message("Test Bell-type state")
bell_ref = None
for i in range(N):
    circuit = H(i)
    for j in range(N):
        if i != j:
            circuit = circuit | CNOT(i, j)
    bell = circuit * st0
    if i == 0:
        bell_ref = bell
    else:
        check_same(bell, bell_ref)

g.message("State:")
g.message(bell)

for i in range(100):
    measured = M() * bell
    for j in range(1, N):
        assert measured.classical_bit[j] == measured.classical_bit[0]
