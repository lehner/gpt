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
import cgpt
import gpt
import numpy as np
from gpt.core.foundation import tensor as foundation, base as foundation_base


def get_mt_entry(self_otype, other_otype):
    self_tag = self_otype.__name__
    other_tag = other_otype.__name__
    if other_tag in self_otype.mtab:
        return self_otype.mtab[other_tag]
    elif self_tag in other_otype.rmtab:
        return other_otype.rmtab[self_tag]
    return None


class tensor(foundation_base):
    foundation = foundation

    def __init__(self, first, second=None):
        if second is not None:
            array, otype = first, second
            if isinstance(otype, str):
                otype = gpt.str_to_otype(otype)
        else:
            otype = first
            array = np.zeros(otype.shape, dtype=np.complex128)

        # allow to match compatible shapes
        if array.shape != otype.shape:
            array = np.reshape(array, otype.shape)

        self.array = np.ascontiguousarray(array)
        self.otype = otype
        assert self.array.shape == otype.shape

    def __repr__(self):
        return "tensor(%s,%s)" % (str(self.array), self.otype.__name__)

    def describe(self):
        return self.otype.__name__

    def __getitem__(self, a):
        return self.array.__getitem__(a)

    def __setitem__(self, a, b):
        return self.array.__setitem__(a, b)

    def nfloats(self):
        return self.otype.nfloats

    def transposable(self):
        return self.otype.transposed is not None

    def transpose(self):
        if not self.transposable():
            return gpt.transpose(gpt.expr(self))
        return tensor(np.transpose(self.array, self.otype.transposed), self.otype)

    def conj(self):
        return tensor(self.array.conj(), self.otype)

    def copy(self):
        return tensor(np.copy(self.array), self.otype)

    def new(self):
        return tensor(np.zeros(shape=self.array.shape, dtype=self.array.dtype), self.otype)

    def adj(self):
        if not self.transposable():
            return gpt.adj(gpt.expr(self))
        return tensor(np.transpose(self.array.conj(), self.otype.transposed), self.otype)

    def reduced(self):
        if self.otype.data_otype() == gpt.ot_singlet:
            return complex(self.array)
        return self

    def trace(self, t):
        res = self
        if t & gpt.expr_unary.BIT_SPINTRACE:
            st = res.otype.spintrace
            assert st is not None and len(st) == 3  # do not yet support tracing vectors
            if st[0] is not None:
                res = tensor(np.trace(res.array, offset=0, axis1=st[0], axis2=st[1]), st[2]())
        if t & gpt.expr_unary.BIT_COLORTRACE:
            ct = res.otype.colortrace
            assert ct is not None and len(ct) == 3
            if ct[0] is not None:
                res = tensor(np.trace(res.array, offset=0, axis1=ct[0], axis2=ct[1]), ct[2]())

        if isinstance(res.otype, gpt.ot_singlet):
            res = complex(res.array)
        return res

    def norm2(self):
        return np.linalg.norm(self.array) ** 2.0

    def __mul__(self, other):
        if isinstance(other, gpt.tensor):
            mt = get_mt_entry(self.otype, other.otype)
            if mt is None:
                mt = get_mt_entry(self.otype.data_otype(), other.otype.data_otype())
                assert mt is not None
            a = np.tensordot(self.array, other.array, axes=mt[1])
            if len(mt) > 2:
                a = np.transpose(a, mt[2])
            return tensor(a, mt[0]())
        elif gpt.util.is_num(other):
            return tensor(self.array * complex(other), self.otype)
        elif isinstance(other, gpt.expr) and other.is_single(gpt.tensor):
            ue, uf, to = other.get_single()
            if ue == 0 and uf & gpt.factor_unary.BIT_TRANS != 0:
                tag = to.otype.__name__
                assert tag in self.otype.otab
                mt = self.otype.otab[tag]
                rhs = to.array
                if uf & gpt.factor_unary.BIT_CONJ != 0:
                    rhs = rhs.conj()
                x = np.multiply.outer(self.array, rhs)
                for swp in mt[1]:
                    x = np.swapaxes(x, swp[0], swp[1])
                return tensor(x, mt[0]())
            assert 0
        else:
            return other.__rmul__(self)

    def __rmul__(self, other):
        if gpt.util.is_num(other):
            return tensor(self.array * other, self.otype)
        else:
            return other.__mul__(self)

    def __add__(self, other):
        assert self.otype.__name__ == other.otype.__name__
        return tensor(self.array + other.array, self.otype)

    def __truediv__(self, other):
        return tensor(self.array / other, self.otype)

    def __neg__(self):
        return tensor(-self.array, self.otype)

    def __sub__(self, other):
        assert self.otype.__name__ == other.otype.__name__
        return tensor(self.array - other.array, self.otype)

    def __iadd__(self, other):
        assert self.otype.__name__ == other.otype.__name__
        self.array += other.array
        return self

    def __isub__(self, other):
        assert self.otype.__name__ == other.otype.__name__
        self.array -= other.array
        return self

    def __itruediv__(self, other):
        self.array /= other
        return self

    def __imatmul__(self, other):
        assert self.otype.__name__ == other.otype.__name__
        self.array = other.array.copy()
        return self
