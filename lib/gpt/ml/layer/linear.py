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


cache = {}
def projector_color_trace(a, b):
    cache_tag = f"{a.otype.__name__}_{a.grid}_{b.otype.__name__}_{b.grid}"

    if cache_tag not in cache:
        ti = g.stencil.tensor_instructions
        Ns = a.otype.spin_ndim
        Nc = a.otype.color_ndim
        code = []
        for spin1 in range(Ns):
            for spin2 in range(Ns):
                for color in range(Nc):
                    aa = spin1 * Nc + color
                    bb = spin2 * Nc + color
                    dst = spin1 * Ns + spin2
                    code.append((0, dst, ti.mov_cc if color == 0 else ti.inc_cc, 1.0, [(2, 0, bb), (1, 0, aa)]))

        res = g(g.color_trace(a * g.adj(b)))
        res2 = g.lattice(res)
        segments = [(len(code) // (Ns * Ns), Ns * Ns)]
        ein = g.stencil.tensor(res2, [(0, 0, 0, 0)], code, segments)
        ein(res2, a, b)

        eps2 = g.norm2(res - res2) / g.norm2(res)
        assert eps2 < 1e-10

        cache[cache_tag] = (res2, ein)

    res2, ein = cache[cache_tag]
    res3 = g.lattice(res2)
    ein(res3, a, b)
    return res3

class linear(base):
    def __init__(
        self,
        data_grid,
        ot_input,
        ot_weights,
        n_input,
        n_output,
        min_weight_dim=8,
        projector=projector_color_trace,
    ):
        nweights = n_output * n_input

        self.projector = projector

        wdim = min_weight_dim
        while wdim < nweights:
            wdim *= 2

        self.weight_grid = g.grid([wdim], g.double)

        super().__init__(self.weight_grid, ot_weights, ot_weights, 1)

        self.data_grid = data_grid
        self.ot_data = ot_input
        self.n_output = n_output
        self.n_input = n_input

        self.access_cache = {}

    def _get_weight_list(self, weights):
        assert len(weights) == 1
        n = self.n_input
        wall = weights[0][(slice(0, n * self.n_output),), self.access_cache]
        return [
            [g.tensor(wall[n * i + j], self.ot_weights).reduced() for j in range(n)]
            for i in range(self.n_output)
        ]

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
        t = g.timer("linear")
        t("weights")
        layer_input = g.util.to_list(layer_input)
        w = self._get_weight_list(weights)
        t("contract")
        x = self._contract(w, layer_input)
        t()
        if g.default.is_verbose("linear_performance"):
            g.message(t)
        return x

    def projected_gradient_adj(self, weights, layer_input, left):
        layer_input = g.util.to_list(layer_input)
        left = g.util.to_list(left)

        assert len(weights) == 1

        t = g.timer("linear.projected_gradient_adj")
        t("weight list")
        w = self._get_weight_list(weights)
        t("field list")

        t()
        n = self.n_input

        o = g.group.cartesian(weights[0])
        o[:] = 0

        for i in range(len(left)):
            for j in range(n):
                t("sums")
                ip_left_f = g.sum(self.projector(left[i], layer_input[j]))
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

            for l in range(len(left)):
                dinput[i] += g.adj(w[l][i]) * left[l]

        t()

        if g.default.is_verbose("linear_performance"):
            g.message(t)

        return [o, dinput]
