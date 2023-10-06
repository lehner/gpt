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
from gpt.ml.layer import base


def projector_color_trace(x):
    return g.color_trace(x)


class parallel_transport_convolution(base):
    def __init__(
        self,
        data_grid,
        U,
        paths,
        ot_input,
        ot_weights,
        n_input,
        n_output,
        min_weight_dim=8,
        projector=projector_color_trace,
    ):
        nweights = (len(paths) + 1) * n_output * n_input

        self.projector = projector

        wdim = min_weight_dim
        while wdim < nweights:
            wdim *= 2

        self.weight_grid = g.grid([wdim], g.double)

        super().__init__(self.weight_grid, ot_weights, ot_weights, 1)

        self.data_grid = data_grid
        self.ot_data = ot_input
        self.paths = paths
        self.U = U
        self.n_output = n_output
        self.n_input = n_input

        self.access_cache = {}

        tmp = [g.lattice(data_grid, ot_input) for i in range(n_input)]
        self.transport = g.parallel_transport(self.U, paths, tmp)
        self.itransport = None

    def _get_weight_list(self, weights):
        assert len(weights) == 1
        n = (len(self.paths) + 1) * self.n_input
        wall = weights[0][(slice(0, n * self.n_output),), self.access_cache]
        return [
            [g.tensor(wall[n * i + j], self.ot_weights).reduced() for j in range(n)]
            for i in range(self.n_output)
        ]

    def _get_field_list(self, layer_input, ttr):
        layer_input = g.util.to_list(layer_input)

        tr = list(ttr(self.U, layer_input))
        n = len(self.paths)

        ret_f = []
        for j in range(len(layer_input)):
            ret_f.append(layer_input[j])
            for l in range(n):
                ret_f.append(g(tr[l][0] * tr[l][1][j]))
        return ret_f

    def _contract(self, w, f):
        ret = []
        for i in range(len(w)):
            s = g.lattice(self.data_grid, self.ot_data)
            s[:] = 0

            assert len(w[i]) == len(f)
            for w_i, f_i in zip(w[i], f):
                s += w_i * f_i

            ret.append(s)

        if len(ret) == 1:
            return ret[0]
        return ret

    def __call__(self, weights, layer_input):
        w = self._get_weight_list(weights)
        f = self._get_field_list(layer_input, self.transport)
        return self._contract(w, f)

    def projected_gradient_adj(self, weights, layer_input, left):
        left = g.util.to_list(left)

        assert len(weights) == 1

        t = g.timer("projected_gradient_adj")
        t("weight list")
        w = self._get_weight_list(weights)
        t("field list")
        f = self._get_field_list(layer_input, self.transport)

        if self.itransport is None:
            self.itransport = g.parallel_transport(self.U, [p.inverse() for p in self.paths], left)

        t("inverse field list")
        ileft = self._get_field_list(left, self.itransport)

        t()
        n = (len(self.paths) + 1) * self.n_input

        o = g.group.cartesian(weights[0])
        o[:] = 0

        for i in range(len(left)):
            for j in range(n):
                t("sums")
                ip_left_f = g.sum(self.projector(left[i] * g.adj(f[j])))
                pos = n * i + j
                if pos not in self.access_cache:
                    self.access_cache[pos] = {}
                t("sets")
                o[(pos,), self.access_cache[pos]] = ip_left_f
                t()

        t("accumulate")
        dinput = [g.lattice(self.data_grid, self.ot_data) for i in range(self.n_input)]
        for i in range(self.n_input):
            dinput[i][:] = 0

            npath = len(self.paths) + 1

            for l in range(len(left)):
                for j in range(npath):
                    dinput[i] += g.adj(w[l][i * npath + j]) * ileft[l * npath + j]

        t()

        if g.default.is_verbose("parallel_transport_convolution_performance"):
            g.message(t)

        return [o, dinput]
