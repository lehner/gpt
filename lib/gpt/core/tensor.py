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

class tensor:

    def __init__(self, array, otype):
        self.array = array
        self.otype = otype
        assert(self.array.shape == otype.shape)

    def __repr__(self):
        return "tensor(%s,%s)" % (str(self.array),self.otype.__name__)

    def __getitem__(self, a):
        return self.array.__getitem__(a)

    def __setitem__(self, a, b):
        return self.array.__setitem__(a,b)

    def transposable(self):
        return not (self.otype.transposed is None)

    def transpose(self):
        if not self.transposable():
            return gpt.transpose( gpt.expr(self) )
        return tensor( np.transpose(self.array, self.otype.transposed), self.otype )

    def conj(self):
        return tensor( self.array.conj(), self.otype )

    def adj(self):
        if not self.transposable():
            return gpt.adj( gpt.expr(self) )
        return tensor( np.transpose(self.array.conj(), self.otype.transposed), self.otype )

    def trace(self, t):
        res = self
        if (t & gpt.expr_unary.BIT_SPINTRACE):
            st=res.otype.spintrace
            assert(not st is None)
            if not st[0] is None:
                res= tensor( np.trace( res.array, offset = 0, axis1 = st[0], axis2 = st[1]), st[2] )
        if (t & gpt.expr_unary.BIT_COLORTRACE):
            ct=res.otype.colortrace
            assert(not ct is None)
            if not ct[0] is None:
                #res= tensor( np.trace( res.array, offset = 0, axis1 = ct[0], axis2 = ct[1]), ct[2] )
                tr=np.trace( res.array, offset = 0, axis1 = ct[0], axis2 = ct[1])
                res=tensor(np.reshape(tr,(1,)), ct[2])
        if res.otype == gpt.ot_complex:
            res = complex(res.array)
        return res

    def norm2(self):
        return np.linalg.norm(self.array) ** 2.0

    def __mul__(self, other):
        if type(other) == gpt.tensor:
            tag = (self.otype,other.otype)
            assert(tag in gpt.otype.mtab)
            mt=gpt.otype.mtab[tag]
            return tensor( np.tensordot(self.array, other.array, axes = mt[1]), mt[0])
        elif type(other) == complex:
            return tensor( self.array * other, self.otype )
        elif type(other) == gpt.expr and other.is_single(gpt.tensor):
            ue,uf,to=other.get_single()
            if ue == 0 and uf & gpt.factor_unary.BIT_TRANS != 0:
                tag = (self.otype,to.otype)
                assert(tag in gpt.otype.otab)
                mt=gpt.otype.otab[tag]
                rhs=to.array
                if uf & gpt.factor_unary.BIT_CONJ != 0:
                    rhs=rhs.conj()
                x=np.multiply.outer(self.array,rhs)
                for swp in mt[1]:
                    x=np.swapaxes(x,swp[0],swp[1])
                return tensor( x, mt[0])
            assert(0)
        else:
            return other.__rmul__(self)

    def __rmul__(self, other):
        if type(other) == complex:
            return tensor( self.array * other, self.otype )
        else:
            return other.__mul__(self)

    def __add__(self, other):
        assert(self.otype == other.otype)
        return tensor( self.array + other.array, self.otype)

    def __sub__(self, other):
        assert(self.otype == other.otype)
        return tensor( self.array - other.array, self.otype)

    def __iadd__(self, other):
        assert(self.otype == other.otype)
        self.array += other.array
        return self

    def __isub__(self, other):
        assert(self.otype == other.otype)
        self.array -= other.array
        return self

    def __itruediv__(self, other):
        assert(type(other) == complex)
        self.array /= other
        return self
