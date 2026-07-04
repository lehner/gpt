#
#    GPT - Grid Python Toolkit
#    Copyright (C) 2025-26  Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)
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
import cgpt
import numpy as np


def rank_fft(self, source, target, sign):
    # target_{*x} = e^{sign i (2pi/L) n x} source_{*n}
    # no normalization, acts on fastest running dimension
    assert source.shape == target.shape
    assert source.dtype is target.dtype
    howmany = int(np.prod(source.shape[0:-1]))
    size = source.shape[-1]
    cgpt.kernel_fft(self.obj, source.view, target.view, target.dtype, howmany, size, sign)
    return self


def fft(self, source, target, dimension, forward=True):
    # will match gpt convention of having forward transformation with +1 sign
    # need function to transpose one dimension to fastest running one, then expand to global and to undo this
    return self
