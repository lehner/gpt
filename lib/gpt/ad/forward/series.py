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
        other = series({other: 1}, landau_O)
    elif g.util.is_num(other):
        other = series({infinitesimal({}): other}, landau_O)
    return other


class series(base):
    foundation = foundation

    def __init__(self, terms, landau_O):
        self.landau_O = landau_O
        if not isinstance(terms, dict):
            i0 = infinitesimal({})
            terms = {i0: terms}
        self.terms = terms

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
        res = series({i0: functional(root, 0)}, self.landau_O)
        delta = nilpotent
        nfac = 1.0
        for i in range(1, maxn):
            nfac *= i
            res += delta * functional(root, i) / nfac
            if i != maxn - 1:
                delta = delta * nilpotent
        return res

    def __iadd__(self, other):
        res = self + other
        self.landau_O = res.landau_O
        self.terms = res.terms
        return self

    def __mul__(self, other):
        return self.distribute2(other, lambda a, b: a * b)

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
                terms[t2] = other.terms[t2]
            else:
                terms[t2] = g(terms[t2] - other.terms[t2])
        return series(terms, landau_O)

    def __truediv__(self, other):
        return (1.0 / other) * self

    def __radd__(self, other):
        return self.__add__(other, self)

    def __getitem__(self, tag):
        if tag == 1:
            tag = infinitesimal({})
        return self.terms[tag]

    def __setitem__(self, tag, value):
        if tag == 1:
            tag = infinitesimal({})
        self.terms[tag] = value
