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


class parallel_transport:
    def __init__(
        self,
        data_grid,
        U,
        paths,
        ot_input,
    ):
        self.data_grid = data_grid
        self.ot_data = ot_input
        self.paths = paths
        self.U = U
        self.n_weights = 0
        self.o_weights = []

        self.access_cache = {}

        tmp = [g.lattice(data_grid, ot_input) for i in range(len(paths))]
        self.transport = [
            g.parallel_transport(self.U, [p], [t]) for t, p in zip(tmp, paths)
        ]
        self.itransport = None

    def weights(self):
        return self.o_weights

    def _get_field_list(self, layer_input, ttr):
        layer_input = g.util.to_list(layer_input)

        assert len(layer_input) == 1 + len(self.paths)
        assert len(ttr) == len(self.paths)

        ret_f = [layer_input[0]]

        for ttrl, l in zip(ttr, layer_input[1:]):
            xx = list(ttrl(self.U, [l]))
            assert len(xx) == 1
            U_path, l_path = xx[0]
            assert len(l_path) == 1
            ret_f.append(g(U_path * l_path[0]))

        return ret_f

    def __call__(self, weights, layer_input):
        t = g.timer("parallel_transport")
        t("get fields")
        x = self._get_field_list(layer_input, self.transport)
        t()
        if g.default.is_verbose("parallel_transport_performance"):
            g.message(t)
        return x

    def projected_gradient_adj(self, weights, layer_input, left):
        left = g.util.to_list(left)
        layer_input = g.util.to_list(layer_input)

        assert len(weights) == 0
        assert len(left) == len(layer_input)
        assert len(left) == 1 + len(self.paths)

        t = g.timer("parallel_transport.projected_gradient_adj")
        t("field list")
        if self.itransport is None:
            self.itransport = [
                g.parallel_transport(self.U, [p.inverse()], [l])
                for p, l in zip(self.paths, left)
            ]

        t("inverse field list")
        ileft = self._get_field_list(left, self.itransport)

        t()

        if g.default.is_verbose("parallel_transport_performance"):
            g.message(t)
            
        return [ileft]
