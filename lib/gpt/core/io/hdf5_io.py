#
#    GPT - Grid Python Toolkit
#    Copyright (C) 2025  Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)
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
import numpy as np
import os

try:
    import h5py
except ImportError:
    h5py = None

verbose = g.default.is_verbose("io")

default_type_map = {
    (np.complex128, (4, 4, 3, 3)): (
        lambda shape: g.mspincolor(g.grid(shape, g.double)),
        lambda x: x,
    ),
    (np.complex64, (4, 4, 3, 3)): (
        lambda shape: g.mspincolor(g.grid(shape, g.single)),
        lambda x: x,
    ),
}


def load(file, params):

    # check file format
    if not os.path.isfile(file) or open(file, "rb").read(4)[1:4] != "HDF".encode("utf-8"):
        raise NotImplementedError()

    # dependency check
    if h5py is None:
        raise Exception("Loading HDF5 file is not possible without installing the h5py module")

    # type map
    if "type_map" in params:
        type_map = params["type_map"]
    else:
        type_map = default_type_map

    # paths
    if "paths" not in params:
        raise Exception("Need to specify one or more paths inside of the HDF5 file to load")

    paths = params["paths"]

    # file
    f = h5py.File(file, "r")

    ret = []
    for tag in paths:
        path = [x for x in tag.split("/") if x.strip() != ""]
        context = f
        for p in path:
            if p not in context:
                raise Exception(f"{tag} not found in {file}")
            context = context[p]

        # space-time shape
        grid_shape = context.shape

        # find precision and internal shape
        dtype = context.dtype
        internal_shape = []
        while dtype.subdtype is not None:
            internal_shape = internal_shape + list(dtype.shape)
            dtype = dtype.base
        dtype = dtype.type
        internal_shape = tuple(internal_shape)

        # find type map
        type_key = (dtype, internal_shape)
        if type_key not in type_map:
            raise Exception(f"{type_key} not in type_map")
        creator, munger = type_map[type_key]

        # now create target
        lattice = creator(grid_shape)
        grid = lattice.grid

        # create access slice
        access = []
        for mu in range(len(grid_shape)):
            x0 = grid.processor_coor[mu] * grid.ldimensions[mu]
            x1 = x0 + grid.ldimensions[mu]
            access.append(slice(x0, x1))

        # load local data
        g.barrier()
        t0 = g.time()
        local_data = context[tuple(access)]
        for mu in range(len(grid_shape) // 2):
            local_data = local_data.swapaxes(mu, len(grid_shape) - mu - 1)

        if verbose:
            nbytes = int(local_data.nbytes)
            nbytes = grid.globalsum(nbytes)
            t1 = g.time()

            g.message(
                f"Read {tag} from {file} with total of {nbytes / 1e9:.2g} GB at {nbytes / 1e9 / (t1 - t0):.2g} GB/s"
            )

        # set data
        lattice[:] = 0
        lattice[:] = np.ascontiguousarray(munger(local_data).astype(dtype))

        if len(paths) == 1:
            return lattice

        ret.append(lattice)

    return ret
