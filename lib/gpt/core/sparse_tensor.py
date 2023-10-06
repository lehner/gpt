#
#    GPT - Grid Python Toolkit
#    Copyright (C) 2022  Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)
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
import gpt
import cgpt


class basis:
    def __init__(self, arg):
        if isinstance(arg, int):
            self.obj = arg
        else:
            self.obj = cgpt.create_tensor_basis(arg)

    def __del__(self):
        cgpt.delete_tensor_basis(self.obj)

    def __getitem__(self, key):
        return cgpt.tensor_basis_get(self.obj, key)

    def __len__(self):
        return cgpt.tensor_basis_get(self.obj, None)

    def to_array(self):
        return [self[i] for i in range(len(self))]

    def __str__(self):
        return str(self.to_array())


class tensor:
    def __init__(self, b, n_parallel, obj=None):
        assert isinstance(b, basis)
        self.basis = b
        if obj is None:
            # a sparse tensor object always needs a dense  vector index
            obj = cgpt.create_sparse_tensor(b.obj, n_parallel)
        self.obj = obj
        self.n_parallel = n_parallel

    def __del__(self):
        cgpt.delete_sparse_tensor(self.obj)

    def update(self, v):
        cgpt.sparse_tensor_set(self.obj, v)

    def __setitem__(self, key, value):
        if key == slice(None, None, None):
            self.update(value)
        else:
            if not isinstance(value, list):
                value = [value] * self.n_parallel
            if isinstance(key, int):
                key = (key,)
            self.update([{key: complex(v)} for v in value])

    def __getitem__(self, key):
        if key == slice(None, None, None):
            key = None
        if isinstance(key, int):
            key = (key,)
        return cgpt.sparse_tensor_get(self.obj, key)

    def __mul__(self, other):
        return self.binary(other, 1)

    def __rmul__(self, other):
        if gpt.util.is_num(other):
            return self * other
        else:
            raise NotImplementedError()

    def __add__(self, other):
        return self.binary(other, 0)

    def __sub__(self, other):
        return self + other * (-1)

    def sum(self):
        obj_t, obj_b = cgpt.sparse_tensor_sum(self.obj)
        return tensor(basis(obj_b), 1, obj_t)

    def tensor_remove(self):
        assert self.n_parallel == 1
        if len(self.basis) == 0:
            return self[()][0]
        else:
            return self

    def global_sum(self):
        me = str([self.basis.to_array(), self[:]]).encode("utf-8")

        r = tensor(basis([]), self.n_parallel)

        for i in range(gpt.ranks()):
            b, t = eval(gpt.broadcast(i, me).decode("utf-8"))
            ti = tensor(basis(b), self.n_parallel)
            ti[:] = t
            r = r + ti

        return r

    def binary(self, other, l):
        obj_t, obj_b = cgpt.sparse_tensor_binary(
            self.obj, other.obj if isinstance(other, tensor) else complex(other), l
        )
        return tensor(basis(obj_b), self.n_parallel, obj_t)

    def __str__(self):
        return str(self[:])


def contract(tensors, symbols):
    obj_t, obj_b = cgpt.sparse_tensor_contract([t.obj for t in tensors], [s[0] for s in symbols])
    return tensor(basis(obj_b), tensors[0].n_parallel, obj_t)
