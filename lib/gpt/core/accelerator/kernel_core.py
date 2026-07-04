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


def accumulate(self, buffers, scales=None, zero=False):
    assert all([buffers[0].shape == b.shape for b in buffers])
    assert all([buffers[0].dtype == b.dtype for b in buffers])
    if scales is None:
        scales = np.array([1] * (len(buffers) - 1), dtype=buffers[0].dtype)
    assert isinstance(scales, np.ndarray)
    assert scales.shape == (len(buffers) - 1,)
    assert scales.dtype == buffers[0].dtype
    self.references.append([x.view for x in buffers])
    cgpt.kernel_accumulate(
        self.obj,
        int(np.prod(buffers[0].shape)),
        [x.view for x in buffers],
        buffers[0].dtype,
        scales, 1 if zero else 0
    )
    return self


def indexed_sum(self, source, index, target, accumulate=False):
    assert source.dtype is target.dtype
    assert index.dtype is np.int64
    assert len(target.shape) == 1
    assert len(index.shape) <= len(source.shape)
    self.references.append(source)
    self.references.append(index)
    self.references.append(target)
    for i in range(len(index.shape)):
        assert index.shape[i] == source.shape[i]
    cgpt.kernel_indexed_sum(
        self.obj,
        source.view,
        source.shape,
        index.view,
        len(index.shape),
        target.view,
        target.shape[0],
        source.dtype,
        1 if accumulate else 0,
    )
    return self


def transpose(self, dst, src, axes):
    axes = tuple(axes)
    assert len(axes) == len(src.shape) and len(src.shape) == len(dst.shape)
    assert set(axes) == set(range(len(axes)))
    assert dst.shape == tuple(src.shape[i] for i in axes)
    assert dst.dtype is src.dtype
    self.references.append(src)
    self.references.append(dst)
    cgpt.kernel_transpose_device_memory_view(self.obj, dst.view, src.view, src.shape, axes)
    return self


def contract(self, *code):
    assert all(isinstance(x, (list, tuple)) for x in code)
    assert all(isinstance(x[0], g.accelerator.buffer) for x in code)
    assert all(isinstance(y, str) for x in code for y in x[1:])

    # add references so that memory used will not be deallocated
    self.references.append([x[0].view for x in code])

    tensors = [x[0] for x in code]
    dtype = tensors[0].dtype
    assert all(t.dtype is dtype for t in tensors[1:])

    tags = {}
    dimensions = []
    conjugate = []
    for t in range(len(tensors)):
        conjugate.append("*" in code[t])
        indices = tuple(x for x in code[t][1:] if x != "*")
        assert len(indices) == len(tensors[t].shape)
        for d in range(len(indices)):
            if indices[d] not in tags:
                nd = tensors[t].shape[d]
                tags[indices[d]] = (len(dimensions), nd)
                dimensions.append(nd)
            else:
                assert tags[indices[d]][1] == tensors[t].shape[d]

    # now construct strides
    strides = [[0] * len(dimensions) for t in tensors]
    for t in range(len(tensors)):
        indices = tuple(x for x in code[t][1:] if x != "*")
        tstrides = [int(np.prod(tensors[t].shape[i + 1 :])) for i in range(len(indices))]
        for d in range(len(indices)):
            strides[t][tags[indices[d]][0]] += tstrides[d]
    cgpt.kernel_contract(self.obj, [x.view for x in tensors], strides, dimensions, conjugate, dtype)

    # TODO: need multi_contract version which can do multiple at the same time if they
    # share parallel indices
    return self
