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
r = g.random("qis_test")

# tests done for N qubits
N = 11
g.message(f"Run tests with {N} qubits")


# helper functions
def set_value_of_qubit(state, i, val):
    s = state.cloned()
    mv = s.measure(i)
    if mv != val:
        s.X(i)
    return s


# special test for dynamic prefetching
st = g.qis.backends.dynamic.state(r, N, precision=g.double)
for i in range(N):
    st0 = X(i) * st
    st0.prefetch([5, 4, 2])
    st0 = M() * st0
    ref = [0] * N
    ref[i] = 1
    assert st0.classical_bit == ref

# state mangler
mangle = {
    g.qis.backends.static: lambda st: st,
    g.qis.backends.dynamic: lambda st: st.prefetch([8, 7, 3, 6]),
}


# test different backends
for q in [g.qis.backends.static, g.qis.backends.dynamic]:
    g.message(f"Testing {q.__name__}")

    st0 = q.state(r, N, precision=g.double)
    stR = st0.cloned()
    stR.randomize()

    # first make sure state is properly initialized
    g.message("Test initial state")
    st1 = st0.cloned()
    mangle[q](st1)
    psi = M() * st1
    assert psi.classical_bit == [0] * N

    # now test all X gates
    g.message("Test X")
    for i in range(N):
        psi = X(i) * st0
        mangle[q](psi)
        psi2 = M() * psi
        ref = [0] * N
        ref[i] = 1
        assert psi2.classical_bit == ref  # then test coefficients and m

        # test that direct access is compatible
        assert (abs(psi[2**i]) - 1.0) < 1e-7

    # now test X^2 = 1
    g.message("Test X^2")
    for i in range(N):
        psi = X(i) * stR
        mangle[q](psi)
        psi = X(i) * psi
        q.check_same(psi, stR)

    # now test H^2 = 1
    g.message("Test H^2")
    for i in range(N):
        psi = H(i) * stR
        mangle[q](psi)
        psi = H(i) * psi
        q.check_same(psi, stR)

    # now test CNOT^2 = 1
    g.message("Test CNOT^2")
    for i in range(N):
        for j in range(N):
            if i != j:
                psi = CNOT(i, j) * stR
                mangle[q](psi)
                psi = CNOT(i, j) * psi
                q.check_same(psi, stR)

    # now test S^4 = 1
    g.message("Test S^4")
    for i in range(N):
        psi = (R_z(i, np.pi / 2.0) | R_z(i, np.pi / 2.0) | R_z(i, np.pi / 2.0)) * stR
        mangle[q](psi)
        psi = R_z(i, np.pi / 2.0) * psi
        q.check_same(psi, stR)

    # check norm of all gates
    g.message("Test gate unitarity")
    for i in range(N):
        q.check_norm(X(i) * stR)
        q.check_norm(R_z(i, 0.125124) * stR)
        q.check_norm(H(i) * stR)
        for j in range(N):
            if i != j:
                q.check_norm(CNOT(i, j) * stR)

    # H | Z | H = X
    g.message("Test H | Z | H == X")
    for i in range(N):
        a = (H(i) | R_z(i, np.pi)) * stR
        mangle[q](a)
        a = H(i) * a
        b = X(i) * stR
        q.check_same(a, b)

    # CNOT
    g.message("Randomized CNOT test")
    for control in range(N):
        # on a random vector with fixed control qubit value, perform the test
        state_control_0 = set_value_of_qubit(stR, control, 0)
        state_control_1 = set_value_of_qubit(stR, control, 1)
        for target in range(N):
            if target != control:
                psi = CNOT(control, target) * state_control_0
                q.check_same(psi, state_control_0)

                psi = (CNOT(control, target) | X(target)) * state_control_1
                q.check_same(psi, state_control_1)

    # daggers of circuits
    g.message("Dagger of circuit")
    circuit = (
        H(0)
        | R_z(0, np.random.uniform(0, 2 * np.pi))
        | CNOT(0, 1)
        | X(1)
        | R_z(1, np.random.uniform(0, 2 * np.pi))
        | H(1)
        | CNOT(1, 2)
    )
    id_circuit = circuit | circuit.dagger()
    q.check_same(stR, id_circuit * stR)

    # Bell state
    g.message("Test Bell-type state")
    bell_ref = None
    st1 = st0.cloned()
    mangle[q](st1)
    for i in range(N):
        circuit = H(i)
        for j in range(N):
            if i != j:
                circuit |= CNOT(i, j)
        bell = circuit * st1
        if i == 0:
            bell_ref = bell
        else:
            q.check_same(bell, bell_ref)

    g.message("State:")
    g.message(bell)

    for i in range(100):
        measured = M() * bell
        for j in range(1, N):
            assert measured.classical_bit[j] == measured.classical_bit[0]
