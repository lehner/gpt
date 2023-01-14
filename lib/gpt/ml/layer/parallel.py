#
#    GPT - Grid Python Toolkit
#    Copyright (C) 2022  Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)
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
from gpt.ml.layer.group import group


class parallel(group):
    def __init__(self, *layers):
        super().__init__(layers)
        self.left_index = None

    def __call__(self, weights, layer_input):
        ret = []
        left_index = []
        for i in range(len(self.layers)):
            i0 = len(ret)
            ret = ret + g.util.to_list(self.forward(i, weights, layer_input))
            i1 = len(ret)
            left_index.append((i0, i1))
        if self.left_index is None:
            self.left_index = left_index
        return ret

    def projected_gradient_adj(self, weights, input_layer, left):
        r = [None for x in weights]

        is_list = isinstance(input_layer, list)
        l = [g.lattice(x) for x in g.util.to_list(input_layer)]
        for x in l:
            x[:] = 0

        assert self.left_index is not None

        for i in range(len(self.layers)):
            l_i0, l_i1 = self.left_index[i]
            w_i0, w_i1 = self.weights_index[i]

            gr = self.dforward_adj(i, weights, input_layer, left[l_i0:l_i1])

            for j in range(w_i0, w_i1):
                r[j] = gr[j - w_i0]

            dleft = g.util.to_list(gr[-1])
            for a, b in zip(l, dleft):
                a += b

        if not is_list:
            l = l[0]

        return r + [l]
