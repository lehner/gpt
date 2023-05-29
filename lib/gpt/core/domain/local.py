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
    def __init__(self, grid, margin_top, margin_bottom=None, cb=None):
        super().__init__()

        if cb is None:
            cb = g.none

        if margin_bottom is None:
            margin_bottom = margin_top

        self.grid = grid
        self.cb = cb

        dim = grid.nd

        local_grid_padding = [margin_top[i] + margin_bottom[i] for i in range(dim)]
        self.local_grid = grid.split(
            [1] * dim,
            [grid.fdimensions[i] // grid.mpi[i] + local_grid_padding[i] for i in range(dim)],
        )
        self.gcoor_project = g.coordinates(
            (grid, cb), margin_top=margin_top, margin_bottom=margin_bottom
        )
        self.lcoor_project = g.coordinates((self.local_grid, cb))
        top = np.array(margin_top, dtype=np.int32)
        bottom = np.array(self.local_grid.fdimensions, dtype=np.int32) - margin_bottom
        self.bcoor = np.sum(
            np.logical_and(self.lcoor_project >= top, self.lcoor_project < bottom), axis=1
        ) == len(top)
        self.lcoor_promote = self.lcoor_project[self.bcoor]
        self.gcoor_promote = self.gcoor_project[self.bcoor]

    def bulk(self):
        class _bulk_domain(two_grid_base):
            def __init__(me):
                super().__init__()
                me.local_grid = self.local_grid
                me.gcoor = g.coordinates((self.grid, self.cb))
                me.lcoor = self.lcoor_project[self.bcoor]
                assert len(me.gcoor) == len(me.lcoor)

                me.gcoor_project = me.gcoor
                me.gcoor_promote = me.gcoor
                me.lcoor_project = me.lcoor
                me.lcoor_promote = me.lcoor

        return _bulk_domain()

    def margin(self):
        class _margin_domain(two_grid_base):
            def __init__(me):
                super().__init__()
                me.local_grid = self.local_grid

                me.lcoor = self.lcoor_project[~self.bcoor]
                me.gcoor = me.lcoor

                me.gcoor_project = me.gcoor
                me.gcoor_promote = me.gcoor
                me.lcoor_project = me.lcoor
                me.lcoor_promote = me.lcoor

        return _margin_domain()
