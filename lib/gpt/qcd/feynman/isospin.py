#
#    GPT - Grid Python Toolkit
#    Copyright (C) 2026  Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)
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
import numpy as np


# Build everything up from quark field doublet (u, d) and antiquark field doublet (-dbar, ubar)
# Note: latter transforms in complex representation of SU(2) which in pseudo-real SU(2)
# can be obtained by applying i sigma_1  (or equivalently multiplied by another number)
class irrep:
    def __init__(self, I, vec):
        self.vec = vec
        self.I = I

    def __getitem__(self, I3):
        i = int(self.I - I3 + 0.001)
        return self.vec[i]

    def __str__(self):
        r = ""
        n = int(2 * self.I + 1.0001)
        for i in range(n):
            I3 = self.I - i
            if r != "":
                r = r + " + "
            else:
                r = r + "   "
            r = r + f"|{self.I}, {I3}>:\n{self.vec[i]}"
        return r


#
# Compute CGC
#
clebsch_gordan_cache = {}


def clebsch_gordan(I1, I2, Itot):
    tag = (I1, I2, Itot)
    if tag in clebsch_gordan_cache:
        return clebsch_gordan_cache[tag]

    # Need all total I between Itot and I1 + I2 (included) for orthogonalization
    Iprep = I1 + I2
    while Iprep > Itot:
        clebsch_gordan(I1, I2, Iprep)
        Iprep -= 1

    # Compute CGC
    cgc = []

    # Start with highest
    if Itot == I1 + I2:
        cgc.append([(I1, I2, 1)])
    else:
        # need orthogonalization for Itot < I1 + I2
        Ired = int(I1 + I2 - Itot + 0.01)
        # can be in sub-space of (I1 - Ired, I2), (I1 - Ired + 1, I2 - 1), ..., (I1, I2 - Ired)
        subspace = [(I1 - Ired + i, I2 - i) for i in range(Ired + 1)]
        ortho = []
        for i in range(Ired):
            nw = {}
            for m1, m2, c in clebsch_gordan_cache[I1, I2, I1 + I2 - i][Ired - i]:
                nw[m1, m2] = c
            for m1, m2 in subspace:
                if (m1, m2) not in nw:
                    nw[m1, m2] = 0
            ortho.append(np.array([nw[m1, m2] for m1, m2 in subspace], dtype=np.float64))

        nn = 0
        while True:
            v = np.array(
                [np.cos(2 * np.pi * nn * i / len(subspace)) for i in range(len(subspace))],
                dtype=np.float64,
            )
            # now Gram-Schmidt
            for ov in ortho:
                v -= ov * np.dot(v, ov) / np.dot(ov, ov)
            if np.linalg.norm(v) > 1e-4:
                break
            nn += 1
        v /= np.linalg.norm(v)
        if v[-1] < 0:
            v *= -1
        cgc.append([(*subspace[i], float(v[i])) for i in range(len(subspace))])

    # Fill in rest by lowering
    for n in range(int(Itot * 2 + 0.01)):
        nw = {}
        for m1, m2, c in cgc[-1]:
            if (m1 - 1, m2) not in nw:
                nw[m1 - 1, m2] = 0
            if (m1, m2 - 1) not in nw:
                nw[m1, m2 - 1] = 0
            m = m1 + m2
            nrm = (Itot * (Itot + 1) - m * (m - 1)) ** 0.5
            nw[m1 - 1, m2] += (I1 * (I1 + 1) - m1 * (m1 - 1)) ** 0.5 / nrm * c
            nw[m1, m2 - 1] += (I2 * (I2 + 1) - m2 * (m2 - 1)) ** 0.5 / nrm * c
        nw = [(m1, m2, nw[m1, m2]) for m1, m2 in nw if abs(nw[m1, m2]) > 1e-14]
        cgc.append(nw)
    assert len(cgc) == int(Itot * 2 + 1 + 0.01)

    clebsch_gordan_cache[tag] = cgc
    return cgc


#
# Combine isospin
#
def multiplet(a, b, I):
    I1 = a.I
    I2 = b.I
    Itot = I
    cgc = clebsch_gordan(I1, I2, Itot)
    return irrep(Itot, [sum(a[m1] * b[m2] * c for m1, m2, c in cmd) for cmd in cgc])


#
# Fundamental building block: quark and antiquark doublets
#
def quark(arg):
    f = g.qcd.feynman
    return irrep(1 / 2, [f.field("up", arg), f.field("down", arg)])


def antiquark(arg):
    f = g.qcd.feynman
    return irrep(1 / 2, [(-1) * f.field("downbar", arg), f.field("upbar", arg)])


#
# Convenience functions
#
def pions(arg):
    return multiplet(antiquark(arg), quark(arg), I=1)


def eta(arg):
    return 1j * multiplet(antiquark(arg), quark(arg), I=0)[0]


def piP(arg):
    return 1j * pions(arg)[-1]


def piM(arg):
    return 1j * pions(arg)[1]


def pi0(arg):
    return 1j * pions(arg)[0]


def two_pions(arg1, arg2, I):
    return multiplet(pions(arg1), pions(arg2), I)


def pipi(arg1, arg2, I, I3):
    return two_pions(arg1, arg2, I)[I3]
