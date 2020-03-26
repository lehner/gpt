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

class propagator:
    def __init__(self, sc_solver):
        self.sc_solver = sc_solver

    def __call__(self, src, dst):
        grid=src.grid
        # sum_n D^-1 vn vn^dag src = D^-1 vn (src^dag vn)^dag
        dst_sc,src_sc=gpt.vspincolor(grid),gpt.vspincolor(grid)

        for s in range(4):
            for c in range(3):
            
                gpt.qcd.prop_to_ferm(src_sc,src,s,c)

                self.sc_solver(src_sc,dst_sc)

                gpt.qcd.ferm_to_prop(dst,dst_sc,s,c)
