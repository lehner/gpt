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

class g5m_ne:
    def __init__(self, matrix, inverter):
        self.matrix = matrix
        self.inverter = inverter
        self.F_grid=matrix.F_grid
        self.ftmp=gpt.vspincolor(self.F_grid)
        self.ftmp2=gpt.vspincolor(self.F_grid)
        self.ftmp3=gpt.vspincolor(self.F_grid)

    def __call__(self, src_sc, dst_sc):

        self.matrix.ImportPhysicalFermionSource(src_sc, self.ftmp)

        self.ftmp @= gpt.gamma[5] * self.ftmp
        self.matrix.G5M(self.ftmp,self.ftmp2)
        
        self.ftmp[:]=0
        self.inverter(lambda i,o: (self.matrix.G5M(i,self.ftmp3),self.matrix.G5M(self.ftmp3,o)),self.ftmp2,self.ftmp)

        self.matrix.ExportPhysicalFermionSolution(self.ftmp,dst_sc)
