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

import gpt
import cgpt
import numpy
import sys


################################################################################
# Merging / Separating along space-time coordinates
################################################################################
def merge(lattices, dimension=-1, N=-1):
    # if only one lattice is given, return immediately
    if not isinstance(lattices, list):
        return lattices

    # number of lattices
    n = len(lattices)
    assert n > 0

    # number of batches
    if N == -1:
        N = n
    batches = n // N
    assert n % N == 0

    # all grids need to be the same
    grid = lattices[0].grid
    assert all([lattices[i].grid.obj == grid.obj for i in range(1, n)])

    # allow negative indexing
    if dimension < 0:
        dimension += grid.nd + 1
        assert dimension >= 0
    else:
        assert dimension <= grid.nd

    # infer checkerboarding of new dimension
    cb = [x.checkerboard() for x in lattices]
    if cb[0] is gpt.none:
        assert all([x is gpt.none for x in cb[1:]])
        cb_mask = 0
    else:
        assert all(
            [cb[j * N + i] is cb[j * N + i + 1].inv() for i in range(N - 1) for j in range(batches)]
        )
        cb_mask = 1

    # otypes must be consistent
    otype = lattices[0].otype
    assert all([lattices[i].otype.__name__ == otype.__name__ for i in range(1, n)])

    # create merged grid
    merged_grid = grid.inserted_dimension(dimension, N, cb_mask=cb_mask)

    # create merged lattices and set checkerboard
    merged_lattices = [gpt.lattice(merged_grid, otype) for i in range(batches)]
    for x in merged_lattices:
        x.checkerboard(cb[0])

    # coordinates of source lattices
    gcoor_zero = gpt.coordinates(lattices[0])
    gcoor_one = gpt.coordinates(lattices[1]) if N > 1 and cb_mask == 1 else gcoor_zero
    gcoor = [gcoor_zero, gcoor_one]

    # data transfer
    for i in range(N):
        merged_gcoor = cgpt.coordinates_inserted_dimension(gcoor[i % 2], dimension, [i])

        plan = gpt.copy_plan(
            merged_lattices[0],
            lattices[i],
            embed_in_communicator=merged_lattices[0].grid,
        )
        plan.destination += merged_lattices[0].view[merged_gcoor]
        plan.source += lattices[i].view[gcoor[i % 2]]
        plan = plan()

        for j in range(batches):
            plan(merged_lattices[j], lattices[j * N + i])

    # if only one batch, remove list
    if len(merged_lattices) == 1:
        return merged_lattices[0]

    # return
    return merged_lattices


def separate(lattices, dimension=-1, cache=None):
    # expect list below
    if not isinstance(lattices, list):
        lattices = [lattices]

    # evaluate in case it is an expression
    lattices = [gpt.eval(x) for x in lattices]

    # number of batches to separate
    batches = len(lattices)
    assert batches > 0

    # make sure all have the same grid
    grid = lattices[0].grid
    assert all([lattices[i].grid.obj == grid.obj for i in range(1, batches)])

    # allow negative indexing
    if dimension < 0:
        dimension += grid.nd
        assert dimension >= 0
    else:
        assert dimension < grid.nd

    # number of slices (per batch)
    N = grid.fdimensions[dimension]
    n = N * batches

    # all lattices need to have same checkerboard
    cb = lattices[0].checkerboard()
    assert all([lattices[i].checkerboard() is cb for i in range(1, batches)])

    # all lattices need to have same otype
    otype = lattices[0].otype
    assert all([lattices[i].otype.__name__ == otype.__name__ for i in range(1, batches)])

    # create grid with dimension removed
    separated_grid = grid.removed_dimension(dimension)
    cb_mask = grid.cb.cb_mask[dimension]

    # create separate lattices and set their checkerboard
    separated_lattices = [gpt.lattice(separated_grid, otype) for i in range(n)]
    for i, x in enumerate(separated_lattices):
        j = i % N
        if cb_mask == 0 or j % 2 == 0:
            x.checkerboard(cb)
        else:
            x.checkerboard(cb.inv())

    # construct coordinates
    separated_gcoor_zero = gpt.coordinates(separated_lattices[0])
    separated_gcoor_one = (
        gpt.coordinates(separated_lattices[1]) if N > 1 and cb_mask == 1 else separated_gcoor_zero
    )
    separated_gcoor = [separated_gcoor_zero, separated_gcoor_one]

    # move data
    for i in range(N):
        cache_key = (i, N, otype.__name__, cb.__name__, dimension, str(grid))
        if cache is not None and cache_key in cache:
            plan = cache[cache_key]
        else:
            gcoor = cgpt.coordinates_inserted_dimension(separated_gcoor[i % 2], dimension, [i])
            plan = gpt.copy_plan(
                separated_lattices[i], lattices[0], embed_in_communicator=lattices[0].grid
            )
            plan.destination += separated_lattices[i].view[separated_gcoor[i % 2]]
            plan.source += lattices[0].view[gcoor]
            plan = plan()
            if cache is not None:
                cache[cache_key] = plan

        for j in range(batches):
            plan(separated_lattices[j * N + i], lattices[j])

    # return
    return separated_lattices


################################################################################
# Merging / Separating along internal indices
################################################################################
default_merge_indices_cache = {}


def separate_indices(x, st, cache=default_merge_indices_cache):
    pos = gpt.coordinates(x)
    cb = x.checkerboard()
    assert st is not None
    result_otype = st[-1]()
    if result_otype is None:
        return x
    ndim = x.otype.shape[st[0]]
    rank = len(st) - 1
    islice = [slice(None, None, None) for i in range(len(x.otype.shape))]
    ivec = [0] * rank
    result = {}

    keys = []
    tidx = []
    dst = []
    for i in range(ndim**rank):
        idx = i
        for j in range(rank):
            c = idx % ndim
            islice[st[j]] = c
            ivec[j] = c
            idx //= ndim
        keys.append(tuple(ivec))
        tidx.append(tuple(islice))

    for i in keys:
        v = gpt.lattice(x.grid, result_otype)
        v.checkerboard(cb)
        result[i] = v
        dst.append(v)

    cache_key = f"separate_indices_{cb.__name__}_{result_otype.__name__}_{x.otype.__name__}_{x.grid.describe()}"
    if cache_key not in cache:
        plan = gpt.copy_plan(dst, x)
        for i in range(len(tidx)):
            plan.destination += result[keys[i]].view[pos]
            plan.source += x.view[(pos,) + tidx[i]]
        cache[cache_key] = plan()

    cache[cache_key](dst, x)

    return result


def separate_spin(x):
    return separate_indices(x, x.otype.spintrace)


def separate_color(x):
    return separate_indices(x, x.otype.colortrace)


def merge_indices(dst, src, st, cache=default_merge_indices_cache):
    pos = gpt.coordinates(dst)
    assert st is not None
    result_otype = st[-1]()
    if result_otype is None:
        dst @= src
        return
    ndim = dst.otype.shape[st[0]]
    rank = len(st) - 1
    islice = [slice(None, None, None) for i in range(len(dst.otype.shape))]
    ivec = [0] * rank
    cache_key = f"merge_indices_{dst.describe()}_{result_otype.__name__}_{dst.grid.describe()}"

    tidx = []
    src_i = []
    for i in range(ndim**rank):
        idx = i
        for j in range(rank):
            c = idx % ndim
            islice[st[j]] = c
            ivec[j] = c
            idx //= ndim
        src_i.append(src[tuple(ivec)])
        tidx.append(tuple(islice))

    if cache_key not in cache:
        plan = gpt.copy_plan(dst, src_i)
        for i in range(ndim**rank):
            plan.destination += dst.view[(pos,) + tidx[i]]
            plan.source += src_i[i].view[:]
        cache[cache_key] = plan()

    cache[cache_key](dst, src_i)


def merge_spin(dst, src):
    merge_indices(dst, src, dst.otype.spintrace)
    return dst


def merge_color(dst, src):
    merge_indices(dst, src, dst.otype.colortrace)
    return dst
