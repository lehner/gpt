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
from gpt.ml.layer import base_no_bias
from gpt.ml.layer.parallel_transport_convolution import projector_color_trace


class local_parallel_transport_convolution(base_no_bias):
    def __init__(
        self,
        grid,
        U,
        paths,
        ot_input,
        ot_weights,
        n_input,
        n_output,
        projector=projector_color_trace,
    ):
        self.nweights = (len(paths) + 1) * n_output * n_input

        self.projector = projector

        super().__init__(grid, ot_input, ot_weights, self.nweights)

        self.paths = paths
        self.U = U
        self.n_output = n_output
        self.n_input = n_input

        tmp = [g.lattice(grid, ot_input) for i in range(n_input)]
        self.transport = g.parallel_transport(self.U, paths, tmp)
        self.itransport = None

    def _get_weight_list(self, weights):
        assert len(weights) == self.nweights
        n = (len(self.paths) + 1) * self.n_input
        return [[weights[n * i + j] for j in range(n)] for i in range(self.n_output)]

    def _get_field_list(self, layer_input, ttr, local):
        layer_input = g.util.to_list(layer_input)

        tr = list(ttr(self.U, layer_input))

        ret_f = []
        for j in range(len(layer_input)):
            if local:
                ret_f.append(layer_input[j])
            for l in range(len(tr)):
                ret_f.append(g(tr[l][0] * tr[l][1][j]))
        return ret_f

    def _contract(self, w, f):
        ret = []
        for i in range(len(w)):
            s = g.lattice(self.grid, self.ot_input)
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
        f = self._get_field_list(layer_input, self.transport, True)
        return self._contract(w, f)

    def projected_gradient_adj(self, weights, layer_input, left):
        left = g.util.to_list(left)

        t = g.timer("projected_gradient_adj")
        t("weight list")
        w = self._get_weight_list(weights)
        t("field list")
        f = self._get_field_list(layer_input, self.transport, True)

        if self.itransport is None:
            self.itransport = [
                g.parallel_transport(
                    self.U,
                    [p.inverse()],
                    [
                        g.lattice(self.grid, self.ot_input)
                        for i in range(self.n_input * self.n_output)
                    ],
                )
                for p in self.paths
            ]

        t()
        n = (len(self.paths) + 1) * self.n_input

        o = []

        for i in range(len(left)):
            for j in range(n):
                o_n = g.group.cartesian(weights[0])
                t("outer products")
                o_n @= self.projector(left[i] * g.adj(f[j]))
                t()
                o.append(o_n)

        t("inverse field list")
        npath = len(self.paths) + 1
        ileft = [
            g(g.adj(w[l][i * npath]) * left[l])
            for l in range(len(left))
            for i in range(self.n_input)
        ]
        for j in range(1, npath):
            ileft = ileft + self._get_field_list(
                [
                    g(g.adj(w[l][i * npath + j]) * left[l])
                    for l in range(len(left))
                    for i in range(self.n_input)
                ],
                self.itransport[j - 1],
                False,
            )

        t("accumulate")
        dinput = [g.lattice(self.grid, self.ot_input) for i in range(self.n_input)]
        for i in range(self.n_input):
            dinput[i][:] = 0

            for l in range(len(left)):
                for j in range(npath):
                    t("accumulate")
                    dinput[i] += ileft[j * len(left) * self.n_input + l * self.n_input + i]

        t()

        if g.default.is_verbose("local_parallel_transport_convolution_performance"):
            g.message(t)

        return o + [dinput]
