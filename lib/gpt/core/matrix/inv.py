#
#    GPT - Grid Python Toolkit
#    Copyright (C) 2020-22  Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)
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
import gpt, cgpt
import numpy as np


def deflate(A):
    if len(A.shape) == 4:
        n1 = A.shape[0]
        n2 = A.shape[2]
        return n1, n2, np.swapaxes(A, 1, 2).reshape(n1 * n2, n1 * n2)
    return None, None, A


def inflate(n1, n2, A):
    if n1 is None:
        return A
    return np.swapaxes(A.reshape(n1, n2, n1, n2), 1, 2)


def inv(A):
    A = gpt.eval(A)

    if isinstance(A, gpt.tensor):
        n1, n2, a = deflate(A.array)
        a = inflate(n1, n2, np.linalg.inv(a))
        return gpt.tensor(a, A.otype)

    assert isinstance(A, gpt.lattice)

    to_list = gpt.util.to_list

    Al = to_list(A)

    if Al[0].otype.shape == (1,):
        return gpt.component.inv(A)

    A_inv = gpt.lattice(A)
    cgpt.invert_matrix(to_list(A_inv), Al)
    return A_inv
