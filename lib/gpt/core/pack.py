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
import cgpt


class pack:
    def __init__(self, lattices):
        self.lattices = g.util.to_list(lattices)
        self.otype = self.lattices[0].otype
        self.grid = self.lattices[0].grid
        self.words = int(np.prod(self.otype.shape))
        assert all([l.otype.__name__ == self.otype.__name__ for l in self.lattices])
        assert all([l.grid.obj == self.grid.obj for l in self.lattices])

    def rank_bytes(self):
        return sum([l.rank_bytes() for l in self.lattices])

    def buffer_coordinates(self, global_coordinates=True):
        nd = self.grid.nd
        L = [self.grid.fdimensions[i] // self.grid.mpi[i] for i in range(nd)]
        if global_coordinates:
            offset = [self.grid.processor_coor[i] * L[i] for i in range(nd)]
        else:
            offset = [0] * nd
        args = [slice(offset[i], offset[i] + L[i]) for i in reversed(range(nd))]
        assert self.grid.cb.n == 1  # in future sieve out wrong parity instead
        return np.mgrid[tuple(args)].reshape(nd, -1).T

    def buffer_coordinate_indices(self):
        return np.arange(self.grid.gsites // self.grid.Nprocessors)

    def allocate_accelerator_buffer(self):
        return g.accelerator_buffer(
            self.rank_bytes(),
            tuple(
                list(reversed(self.grid.ldimensions))
                + [len(self.lattices)]
                + list(self.otype.shape)
            ),
            self.grid.precision.complex_dtype,
        )

    def to_accelerator_buffer(self, target_buffer=None):
        if target_buffer is None:
            target_buffer = self.allocate_accelerator_buffer()
        self.transfer_accelerator_buffer(target_buffer, True)
        return target_buffer

    def from_accelerator_buffer(self, buffer):
        self.transfer_accelerator_buffer(buffer, False)

    def transfer_accelerator_buffer(self, buffer, export):
        buf = buffer.view
        r = len(self.otype.shape)

        cgpt.lattice_transfer_scalar_device_buffer(
            self.lattices,
            buf,
            r,
            1 if export else 0,
        )
