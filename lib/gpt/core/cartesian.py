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

class cartesian_view:

    def __init__(self, first, second = None, third = None):
        if type(first) == gpt.grid:
            g=first
            rank=g.processor
            gdimensions=g.gdimensions
            if second is None:
                mpi=[ g.gdimensions[i] // g.ldimensions[i] for i in range(len(g.gdimensions)) ]
            else:
                assert(0)
        elif type(second) == str:
            rank=first
            mpi=[ int(x) for x in second.strip("[]").split(",") ]
            gdimensions=third
        else:
            rank, mpi, gdimensions = first, second, third
        assert(len(mpi) == len(gdimensions))
        self.nd=len(mpi)
        self.rank=rank
        self.mpi=mpi
        self.gdimensions=gdimensions
        self.ranks=1
        self.processor_coor=[ 0 ] * self.nd
        self.ldimensions = [ gdimensions[i] // mpi[i] for i in range(self.nd) ]

        for i in range(self.nd):
            assert(gdimensions[i] % mpi[i] == 0)
            self.ranks *= mpi[i]
            self.processor_coor[i] = rank % mpi[i]
            rank = rank // mpi[i]

        if self.rank < 0 or self.rank >= self.ranks:
            self.processor_coor=[ None ] * self.nd
            self.top=[ 0 ] * self.nd
            self.bottom=[ 0 ] * self.nd
        else:
            self.top=[ self.ldimensions[i]*self.processor_coor[i] for i in range(self.nd) ]
            self.bottom=[ self.top[i] + self.ldimensions[i] for i in range(self.nd) ]
    
    def describe(self):
        return str(self.mpi).replace(" ","")

    def optimal_rank_map(self, grid):
        # return which_processor_of_grid_should_do_IO_for_cv_rank[rank]
        # Do timing of distribute
        pass
