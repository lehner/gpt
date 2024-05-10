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
import gpt, cgpt, numpy, sys


def split_key_to_coordinates_and_indices(grid, key):
    nd = grid.nd
    list_types = [numpy.ndarray, list]

    # l[pos]
    if any([isinstance(key, t) for t in list_types]):
        return key, None

    # all other keys need to be a tuple
    assert isinstance(key, tuple)

    # strip positions
    if any([isinstance(key[0], t) for t in list_types]):
        # l[pos,...]
        pos = key[0]
        key = key[1:]

    else:
        # l[x,y,z,...]
        assert len(key) >= nd
        pos = key[:nd]
        key = key[nd:]

    # l[...,tidx]
    if len(key) == 1 and any([isinstance(key[0], t) for t in list_types]):
        tidx = key[0]

    # l[...,i0,i1,...]
    elif key == ():
        tidx = None

    else:
        tidx = key

    return pos, tidx


def map_pos(grid, cb, key):
    # if list, convert to numpy array
    if isinstance(key, list):
        key = numpy.array(key, dtype=numpy.int32)

    # if key is numpy array, no further processing needed
    if isinstance(key, numpy.ndarray):
        return key

    # if not, we expect a tuple of slices
    assert isinstance(key, tuple)

    # slices without specified start/stop corresponds to memory view limitation for this rank
    if all([k == slice(None, None, None) for k in key]):
        # go through gpt.coordinates to use its caching feature
        return gpt.coordinates((grid, cb), order="lexicographic")

    nd = grid.nd
    key = tuple([k if isinstance(k, slice) else slice(k, k + 1) for k in key])
    assert all([k.step is None for k in key])
    top = [
        grid.fdimensions[i] // grid.mpi[i] * grid.processor_coor[i] if k.start is None else k.start
        for i, k in enumerate(key)
    ]
    bottom = [
        grid.fdimensions[i] // grid.mpi[i] * (1 + grid.processor_coor[i])
        if k.stop is None
        else k.stop
        for i, k in enumerate(key)
    ]
    assert all(
        [
            0 <= top[i] and top[i] <= bottom[i] and bottom[i] <= grid.fdimensions[i]
            for i in range(nd)
        ]
    )

    return cgpt.coordinates_from_cartesian_view(
        top, bottom, grid.cb.cb_mask, cb.tag, "lexicographic"
    )


def map_tidx_and_shape(l, key):
    # create shape of combined lattices
    shapes = [x.otype.shape for x in l]
    assert all([shapes[0][1:] == s[1:] for s in shapes[1:]])
    shape = (sum([s[0] for s in shapes]),) + shapes[0][1:]
    nd = len(shape)

    # if key is None, numpy array of all indices
    if key is None:
        tidx = cgpt.coordinates_from_cartesian_view(
            [0] * nd, list(shape), [0] * nd, gpt.none.tag, "reverse_lexicographic"
        )
        return tidx, shape

    # if key is a list, convert to numpy array
    if isinstance(key, list):
        key = numpy.array(key, dtype=numpy.int32)

    # if key is numpy array, no further processing needed
    if isinstance(key, numpy.ndarray):
        # Need to decide how to index tensor indices.  With lexicographical
        return key, (len(key),)

    # if not, we expect a tuple of either coordinates or slices
    assert isinstance(key, tuple)

    # slices
    key = tuple([k if isinstance(k, slice) else slice(k, k + 1) for k in key])
    assert all([k.step is None for k in key])
    top = [0 if k.start is None else k.start for i, k in enumerate(key)]
    bottom = [shape[i] if k.stop is None else k.stop for i, k in enumerate(key)]
    assert all([0 <= top[i] and top[i] <= bottom[i] and bottom[i] <= shape[i] for i in range(nd)])
    tidx = cgpt.coordinates_from_cartesian_view(
        top, bottom, [0] * nd, gpt.none.tag, "reverse_lexicographic"
    )
    shape = tuple([bottom[i] - top[i] for i in range(nd)])
    return tidx, shape


def map_key(target, key):
    # work on list of lattices
    if isinstance(target, gpt.lattice):
        return map_key([target], key)

    # all lattices need to have the same grid and checkerboard
    grid = target[0].grid
    cb = target[0].checkerboard()
    assert all([x.grid is grid for x in target[1:]]) and all(
        [cb.tag == x.checkerboard().tag for x in target[1:]]
    )

    # special case to select all
    if isinstance(key, slice) and key == slice(None, None, None):
        key = tuple([slice(None, None, None) for i in range(grid.nd)])

    # split coordinate and tensor index descriptors
    pos, tidx = split_key_to_coordinates_and_indices(grid, key)

    # map out the positions and tensor indices
    pos = map_pos(grid, cb, pos)
    tidx, shape = map_tidx_and_shape(target, tidx)

    # return triple of positions, tidx, and shape
    return pos, tidx, shape
