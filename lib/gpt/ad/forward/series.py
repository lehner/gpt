#
#    GPT - Grid Python Toolkit
#    Copyright (C) 2023  Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)
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
from gpt.ad.forward import infinitesimal
from gpt.ad.forward import foundation
from gpt.core.foundation import base


def promote(other, landau_O):
    if isinstance(other, infinitesimal):
        return series({other: 1}, landau_O)
    elif isinstance(other, series):
        return other

    return series({infinitesimal({}): other}, landau_O)


class series(base):
    foundation = foundation

    def __init__(self, terms, landau_O):
        self.landau_O = landau_O
        if not isinstance(terms, dict):
            i0 = infinitesimal({})
            terms = {i0: terms}
        self.terms = terms

    def new(self):
        terms = {t1: self.terms[t1].new() for t1 in self.terms}
        return series(terms, self.landau_O)

    def __str__(self):
        r = ""
        for t in self.terms:
            if r != "":
                r = r + " + "
            r = r + "(" + str(self.terms[t]) + ")"
            si = str(t)
            if si != "":
                r = r + "*" + si
        return r

    def distribute2(self, other, functional):
        other = promote(other, self.landau_O)
        # first merge landau_Os
        landau_O = self.landau_O + other.landau_O
        # then merge terms
        terms = {}
        for t1 in self.terms:
            for t2 in other.terms:
                i = t1 * t2
                if not landau_O.accept(i):
                    continue
                if i not in terms:
                    terms[i] = g(functional(self.terms[t1], other.terms[t2]))
                else:
                    terms[i] += functional(self.terms[t1], other.terms[t2])
        return series(terms, landau_O)

    def distribute1(self, functional):
        # then merge terms
        terms = {}
        for t1 in self.terms:
            if t1 not in terms:
                terms[t1] = g(functional(self.terms[t1]))
            else:
                terms[t1] += functional(self.terms[t1])
        return series(terms, self.landau_O)

    def function(self, functional):
        root = self[1]
        # get nilpotent power
        nilpotent = self - root
        maxn = 0
        i0 = infinitesimal({})
        for t in nilpotent.terms:
            if t == i0:
                continue
            n = 1
            tn = t
            while self.landau_O.accept(tn):
                tn = tn * t
                n += 1
            maxn = max([maxn, n])
        return functional(root, nilpotent, maxn)

    def __iadd__(self, other):
        other = promote(other, self.landau_O)
        res = self + other
        self.landau_O = res.landau_O
        self.terms = res.terms
        return self

    def __mul__(self, other):
        return self.distribute2(other, lambda a, b: a * b)

    def __imul__(self, other):
        res = self * other
        self.landau_O = res.landau_O
        self.terms = res.terms
        return self

    def __rmul__(self, other):
        if g.util.is_num(other):
            return self.__mul__(other)
        raise Exception("Not implemented")

    def __add__(self, other):
        other = promote(other, self.landau_O)
        # first merge landau_Os
        landau_O = self.landau_O + other.landau_O
        # then merge terms
        terms = {}
        for t1 in self.terms:
            if not landau_O.accept(t1):
                continue
            terms[t1] = self.terms[t1]
        for t2 in other.terms:
            if not landau_O.accept(t2):
                continue
            if t2 not in terms:
                terms[t2] = other.terms[t2]
            else:
                terms[t2] = g(terms[t2] + other.terms[t2])
        return series(terms, landau_O)

    def __sub__(self, other):
        other = promote(other, self.landau_O)
        # first merge landau_Os
        landau_O = self.landau_O + other.landau_O
        # then merge terms
        terms = {}
        for t1 in self.terms:
            if not landau_O.accept(t1):
                continue
            terms[t1] = self.terms[t1]
        for t2 in other.terms:
            if not landau_O.accept(t2):
                continue
            if t2 not in terms:
                terms[t2] = -other.terms[t2]
            else:
                terms[t2] = g(terms[t2] - other.terms[t2])
        return series(terms, landau_O)

    def __rsub__(self, other):
        other = promote(other, self.landau_O)
        return other - self

    def __neg__(self):
        return (-1.0) * self

    def __truediv__(self, other):
        return (1.0 / other) * self

    def __radd__(self, other):
        other = promote(other, self.landau_O)
        return other + self

    def __getitem__(self, tag):
        if tag == 1:
            tag = infinitesimal({})
        return self.terms[tag]

    def __setitem__(self, tag, value):
        if tag == 1:
            tag = infinitesimal({})
        self.terms[tag] = value

    def get_grid(self):
        for t1 in self.terms:
            return self.terms[t1].grid

    def get_otype(self):
        for t1 in self.terms:
            return self.terms[t1].otype

    def set_otype(self, otype):
        for t1 in self.terms:
            self.terms[t1].otype = otype

    def __imatmul__(self, other):
        assert self.landau_O is other.landau_O
        terms = {}
        for t1 in other.terms:
            terms[t1] = g.copy(other.terms[t1])
        self.terms = terms
        return self

    def get_real(self):
        return self.distribute1(lambda a: a.real)

    grid = property(get_grid)
    real = property(get_real)
    otype = property(get_otype, set_otype)


def make(landau_O, O1, *args):
    x = series(O1, landau_O)
    n = len(args)
    assert n % 2 == 0
    for i in range(n // 2):
        x[args[2 * i + 0]] = args[2 * i + 1]
    return x
