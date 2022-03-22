#
#    GPT - Grid Python Toolkit
#    Copyright (C) 2020  Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)
#
#    This program is free software; you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation; either version 2 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License along
#    with this program; if not, write to the Free Software Foundation, Inc.,
#    51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
#
import gpt as g


def _H(st, i):
    st.H(i)


def _X(st, i):
    st.X(i)


def _R_z(st, i, phase):
    st.R_z(i, phase)


def _CNOT(st, c, t):
    st.CNOT(c, t)


def _M(st, i):
    if i is not None:
        st.measure(i)
    else:
        for i in range(st.number_of_qubits):
            st.measure(i)


class circuit:
    def __init__(self, val=[]):
        self.val = val

    def __or__(self, other):
        return circuit(self.val + other.val)

    def __ior__(self, other):
        self.val += other.val
        return self

    def __mul__(self, original):
        other = original.cloned()
        for op, *args in self.val:
            # t0 = g.time()
            op(other, *args)
            # t1 = g.time()
            # gb = 2 * other.lattice.global_bytes() / 1e9
            # g.message(f"T {gb/(t1-t0)} {op.__name__}")
        return other

    def __len__(self):
        return len(self.val)

    def dagger(self):
        """
        Return the daggered circuit.
        For a circuit ``c`` ``c | c.dagger()`` will be the identity.

        Implementation detail: It's a bit hacky.
        """
        return circuit([ga if ga[0] != _R_z else (*ga[:-1], -ga[-1]) for ga in reversed(self.val)])


def H(i):
    return circuit([(_H, i)])


def X(i):
    return circuit([(_X, i)])


def R_z(i, phase):
    return circuit([(_R_z, i, phase)])


def CNOT(control, target):
    return circuit([(_CNOT, control, target)])


def M(i=None):
    return circuit([(_M, i)])
