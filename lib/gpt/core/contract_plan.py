#
#    GPT - Grid Python Toolkit
#    Copyright (C) 2026  Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)
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


# TODO:
# - add other optimization strategies such as randomly picking the split dimension at each step
class contract_plan:
    def __init__(self, temporary_manager, *code):
        self.temporary_manager = temporary_manager
        assert all(isinstance(x, (list, tuple)) for x in code)
        assert all(isinstance(x[0], g.accelerator_buffer) for x in code)
        assert all(isinstance(y, str) for x in code for y in x[1:])
        tensors = [x[0] for x in code]
        tags = {}
        dimensions = []
        for t in range(len(tensors)):
            indices = tuple(x for x in code[t][1:] if x != "*")
            assert len(indices) == len(tensors[t].shape)
            for d in range(len(indices)):
                if indices[d] not in tags:
                    nd = tensors[t].shape[d]
                    tags[indices[d]] = (len(dimensions), nd)
                    dimensions.append(nd)
                else:
                    assert tags[indices[d]][1] == tensors[t].shape[d]

        self.tags = tags
        self.dimensions = dimensions

        # sorted dimensions
        self.sorted_dimensions = [None] * len(dimensions)
        for t in tags:
            self.sorted_dimensions[tags[t][0]] = t

        # use greedy algorithm
        code_use = code
        codes = []
        temp_release = None
        while True:
            code_def, code_use = self.optimal_split(code_use, temporary_manager)
            if code_def is None:
                break
            if temp_release is not None:
                temporary_manager.release(temp_release)
            temp_release = code_def[0][0]
            codes.append(code_def)
        codes.append(code_use)

        if temp_release is not None:
            temporary_manager.release(temp_release)

        self.code = code
        self.codes = codes

    def optimal_split(self, code, manager):
        best_cost = None
        best_code_def = None
        best_code_use = code
        best_dim = None
        if len(code) <= 3:
            return None, code
        for ud in self.used_dimensions(code):
            code_def, code_use = self.split_code(code, ud, None)
            cost1 = self.code_to_cost(code_def) + self.code_to_cost(code_use)
            if best_cost is None or cost1 < best_cost:
                best_cost = cost1
                best_code_def = code_def
                best_code_use = code_use
                best_dim = ud
        best_code_def, best_code_use = self.split_code(code, best_dim, manager)
        return best_code_def, best_code_use

    def shape_from_dimensions(self, dimensions):
        return tuple(self.tags[x][1] for x in dimensions)

    def split_code(self, code, d, manager):
        ud = self.used_dimensions([x for x in code if d in x])
        ud.remove(d)
        if "*" in ud:
            ud.remove("*")
        if manager is None:
            temp = None
        else:
            temp = manager.request(self.shape_from_dimensions(ud), dtype=code[0][0].dtype)
        code_new = [[temp] + ud]
        code_def = code_new + [x for x in code[1:] if d in x]
        code_use = [code[0]] + [x for x in code[1:] if d not in x] + code_new
        return code_def, code_use

    def code_to_str(self, code):
        return " @ ".join([",".join(x[1:]) for x in code[1:]]) + " -> " + ",".join(code[0][1:])

    def used_dimensions(self, code):
        # preserve order
        ud_set = set(y for x in code for y in x[1:] if y != "*")
        ud = [y for y in self.sorted_dimensions if y in ud_set]
        return ud

    def code_to_cost(self, code):
        return np.prod([self.tags[x][1] for x in self.used_dimensions(code)])

    def __str__(self):
        lines = ""
        total_cst = 0
        orig_cst = self.code_to_cost(self.code)
        code_strs = [self.code_to_str(c) for c in self.codes]
        code_csts = [self.code_to_cost(c) for c in self.codes]
        nm = max(len(s) for s in code_strs)
        for i in range(len(code_csts)):
            total_cst += code_csts[i]
            lines = (
                lines
                + f"{code_strs[i] + ' ' * (nm + 1 - len(code_strs[i]))}   at cost = {code_csts[i]:.2e}\n  "
            )
        return f"""
 Direct contraction:

  {self.code_to_str(self.code)}   at cost = {orig_cst:.2e}

 Optimal contraction path:

  {lines}
 Total cost of optimal path = {total_cst:.2e} or a {orig_cst / total_cst:.2e} speedup
        """

    def __call__(self, blas, optimal=True):
        if optimal:
            # TODO: make blas.contract accept the list directly and perform the operation in a cache-optimal manner
            for c in self.codes:
                blas.contract(*c)
        else:
            blas.contract(*self.code)
        return blas
