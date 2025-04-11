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

        t = g.timer("parallel_transport")

        t("pre-compute")
        # pre-compute PT matrices
        self.matrices = []
        for p in self.paths:
            assert len(p.path) == 1
            dim, disp = p.path[0]
            m = g.identity(self.U[0])
            U_dim = self.U[dim]
            while disp > 0:
                m @= U_dim * g.cshift(m, dim, 1)
                disp -= 1
            while disp < 0:
                m @= g.cshift(g.adj(U_dim) * m, dim, -1)
                disp += 1
            self.matrices.append(m)
        t()

        if g.default.is_verbose("parallel_transport_performance"):
            g.message(t)

    def weights(self):
        return self.o_weights

    def _get_field_list(self, layer_input, forward):
        layer_input = g.util.to_list(layer_input)

        assert len(layer_input) == 1 + len(self.paths)

        ret_f = [layer_input[0]]

        for i, l in enumerate(layer_input[1:]):
            if forward:
                l_path_0 = g.cshift(l, *self.paths[i].path[0])
                ret_f.append(g(self.matrices[i] * l_path_0))
            else:
                l_path_0 = g.cshift(g.adj(self.matrices[i]) * l, *self.paths[i].inverse().path[0])
                ret_f.append(l_path_0)

        return ret_f

    def __call__(self, weights, layer_input):
        t = g.timer("parallel_transport")
        t("get fields")
        x = self._get_field_list(layer_input, True)
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
        t("inverse field list")
        ileft = self._get_field_list(left, False)

        t()

        if g.default.is_verbose("parallel_transport_performance"):
            g.message(t)

        return [ileft]
