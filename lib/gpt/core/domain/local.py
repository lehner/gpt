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
import gpt as g
import numpy as np
from gpt.core.domain.two_grid_base import two_grid_base


class local(two_grid_base):
    def __init__(self, grid, margin, cb=None):
        super().__init__()

        if cb is None:
            cb = g.none

        self.grid = grid
        self.cb = cb

        dim = grid.nd

        self.local_grid = grid.split(
            [1] * dim, [grid.fdimensions[i] + 2 * margin[i] for i in range(dim)]
        )
        self.gcoor = g.coordinates((grid, cb), margin=margin)
        self.lcoor = g.coordinates((self.local_grid, cb))
        top = np.array(margin, dtype=np.int32)
        bottom = np.array(self.local_grid.fdimensions, dtype=np.int32) - top
        self.bcoor = np.sum(np.logical_and(self.lcoor >= top, self.lcoor < bottom), axis=1) == len(
            top
        )

    def bulk(self):
        class _bulk_domain(two_grid_base):
            def __init__(me):
                super().__init__()
                me.local_grid = self.local_grid
                me.gcoor = g.coordinates((self.grid, self.cb))
                me.lcoor = self.lcoor[self.bcoor]
                assert len(me.gcoor) == len(me.lcoor)

        return _bulk_domain()

    def margin(self):
        class _margin_domain(two_grid_base):
            def __init__(me):
                super().__init__()
                me.local_grid = self.local_grid

                me.lcoor = self.lcoor[~self.bcoor]
                me.gcoor = me.lcoor

        return _margin_domain()
