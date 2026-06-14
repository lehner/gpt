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


def contract_plan(temporary_manager, *code):
    return contract_plan_general(temporary_manager, 0, *code)


class contract_plan_general:
    def __init__(self, temporary_manager, nramdom, *code):
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

        # try random optimization parts in addition to greedy
        if nramdom == 0:
            # use greedy algorithm
            codes = self.optimize_greedy(code, temporary_manager)

        else:
            ibest = -1
            cost_greedy = sum(self.code_to_cost(c) for c in self.optimize_greedy(code, None))
            for i in range(nramdom):
                g.default.push_verbose("random", False)
                rng = g.random("random-contraction-" + str(i))
                g.default.pop_verbose()
                cost_random = sum(
                    self.code_to_cost(c) for c in self.optimize_random(code, None, rng)
                )
                if cost_random < cost_greedy:
                    ibest = i

            if ibest == -1:
                codes = self.optimize_greedy(code, temporary_manager)
            else:
                g.default.push_verbose("random", False)
                rng = g.random("random-contraction-" + str(ibest))
                g.default.pop_verbose()
                codes = self.optimize_random(code, temporary_manager, rng)

        self.code = code
        self.codes = codes

    def optimize_general(self, code, temporary_manager, split):
        code_use = code
        codes = []
        temp_release = None
        while True:
            code_def, code_use = split(code_use, temporary_manager)
            if code_def is None:
                break
            if temp_release is not None and temporary_manager is not None:
                temporary_manager.release(temp_release)
            temp_release = code_def[0][0]
            codes.append(code_def)
        codes.append(code_use)

        if temp_release is not None and temporary_manager is not None:
            temporary_manager.release(temp_release)
        return codes

    def optimize_greedy(self, code, temporary_manager):
        return self.optimize_general(code, temporary_manager, lambda c, t: self.optimal_split(c, t))

    def optimize_random(self, code, temporary_manager, rng):
        return self.optimize_general(
            code, temporary_manager, lambda c, t: self.random_split(c, t, rng)
        )

    def random_split(self, code, manager, rng):
        sd = self.splittable_dimensions(code)
        if len(code) <= 3 or len(sd) < 1:
            return None, code
        ud = rng.choice(sd, 1)[0]
        return self.split_code(code, ud, manager)

    def optimal_split(self, code, manager):
        best_cost = None
        best_code_def = None
        best_code_use = code
        best_dim = None
        if len(code) <= 3:
            return None, code
        for ud in self.splittable_dimensions(code):
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

        # cannot split over an index that needs to go into the target
        assert d not in code[0][1:]

        # get a list of all used dimensions in factors that we need for split
        ud = self.used_dimensions([x for x in code if d in x])

        # remove CC and split dimension; this keeps genuine external indices to split
        ud = [x for x in ud if x not in [d, "*"]]

        # re-order participants such that those come first with lower indices w.r.t. initial contraction order
        sort_basis = len(self.sorted_dimensions)
        participants = sorted(
            [c for c in code[1:] if d in c[1:]],
            key=lambda x: sum(
                sort_basis**-i * self.sorted_dimensions.index(y) for i, y in enumerate(x[1:])
            ),
        )

        # other factors that are not part of split are called witnesses
        witnesses = [c for c in code[1:] if d not in c[1:]]

        # create an ordered list of indices for definition of new tensor; prioritize common indices, then in order of participants
        ud_ordered = [x for x in ud if all(x in y[1:] for y in participants)]
        for c in participants:
            ud_new = list((set(c[1:]) & set(ud)) - set(ud_ordered))
            ud_ordered.extend([x for x in ud if x in ud_new])

        # trivial steps
        if manager is None:
            temp = None
        else:
            temp = manager.request(self.shape_from_dimensions(ud_ordered), dtype=code[0][0].dtype)
        code_new = [[temp] + ud_ordered]
        code_def = code_new + participants
        code_use = [code[0]] + witnesses + code_new
        return code_def, code_use

    def code_to_str(self, code):
        return " @ ".join([",".join(x[1:]) for x in code[1:]]) + " -> " + ",".join(code[0][1:])

    def used_dimensions(self, code):
        # preserve order
        ud_set = set(y for x in code for y in x[1:] if y != "*")
        return [y for y in self.sorted_dimensions if y in ud_set]

    def splittable_dimensions(self, code):
        ud = self.used_dimensions(code)
        target_indices = code[0][1:]
        return [x for x in ud if x not in target_indices]

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

    def commit_single_contract(self, blas, code, bm, use_gemm):
        verbose = g.default.is_verbose("contract_plan")

        # can I fall back on faster gemm?
        if len(code) == 3 and use_gemm:
            C = code[0][1:]
            A = code[1][1:]
            B = code[2][1:]
            contraction_indices = (set(A) | set(B)) - set(C)
            if len(contraction_indices) == 1:
                i = contraction_indices.pop()
                # now test that contraction index only appears once in A and B each and split them
                if A.count(i) == 1 and B.count(i) == 1:
                    A_order = [A.index(x) for x in C + [i] if x in A]
                    B_order = [B.index(x) for x in C + [i] if x in B]
                    assert len(A_order) == len(A)
                    assert len(B_order) == len(B)
                    ncommon = len(A) + len(B) - len(C) - 2
                    A_reordered = [A[i] for i in A_order]
                    B_reordered = [B[i] for i in B_order]
                    AB_rest = A_reordered[0:-1] + B_reordered[ncommon:-1]
                    if AB_rest == C:
                        target = code[0][0]
                        A_reshape = tuple(self.tags[i][1] for i in A_reordered)
                        B_reshape = tuple(self.tags[i][1] for i in B_reordered)

                        A_transposed = bm.request(shape=A_reshape, dtype=target.dtype)
                        B_transposed = bm.request(shape=B_reshape, dtype=target.dtype)

                        blas.transpose(A_transposed, code[1][0], A_order)
                        blas.transpose(B_transposed, code[2][0], B_order)

                        A_transposed.reshape(
                            A_reshape[0:ncommon]
                            + (int(np.prod(A_reshape[ncommon:-1])), A_reshape[-1])
                        )
                        B_transposed.reshape(
                            B_reshape[0:ncommon]
                            + (int(np.prod(B_reshape[ncommon:-1])), B_reshape[-1])
                        )

                        target_shape = target.shape
                        target.reshape(
                            A_reshape[0:ncommon]
                            + (
                                int(np.prod(A_reshape[ncommon:-1])),
                                int(np.prod(B_reshape[ncommon:-1])),
                            )
                        )

                        idx = A_transposed.indices(range(ncommon))
                        blas.gemm(1.0, A_transposed[idx], B_transposed[idx].T, 0.0, target[idx])

                        if verbose:
                            g.message(
                                f"Replace contraction by GEMM : {int(np.prod(A_reshape[0:ncommon]))} x {A_transposed.shape[ncommon:]} @ {B_transposed.shape[ncommon:]}.T"
                            )

                        # restore target shape
                        target.reshape(target_shape)

                        bm.release(A_transposed)
                        bm.release(B_transposed)
                        return

        blas.contract(*code)

    def __call__(self, blas, optimal=True, use_gemm=True):
        if optimal:
            bm = g.accelerator_buffer_manager()

            # TODO:
            # Could optimize below for cache re-use by making blas.contract accept the list directly and perform the operation in a cache-optimal manner.
            # Alternative first approach: see if I can map contractions to gemm.  May be fastest in practice!
            for c in self.codes:
                self.commit_single_contract(blas, c, bm, use_gemm)
        else:
            blas.contract(*self.code)
        return blas
