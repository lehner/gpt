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
from gpt.core import auto_tuned_class, auto_tuned_method
import numpy as np
from gpt.core.contract.util import util
from gpt.core.contract.linear_map import linear_map


class contract_plan_general:
    def __init__(self, temporary_manager, nrandom, *code):
        self.temporary_manager = temporary_manager
        self.util = util(code)

        # try random optimization parts in addition to greedy
        if nrandom == 0:
            # use greedy algorithm
            codes = self.optimize_greedy(code, temporary_manager)

        else:
            ibest = -1
            cost_greedy = sum(self.code_to_cost(c) for c in self.optimize_greedy(code, None))
            for i in range(nrandom):
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

    def optimize_greedy(self, code, temporary_manager):
        return self.util.optimize_general(
            code, temporary_manager, lambda c, t: self.optimal_split(c, t)
        )

    def optimize_random(self, code, temporary_manager, rng):
        return self.util.optimize_general(
            code, temporary_manager, lambda c, t: self.random_split(c, t, rng)
        )

    def random_split(self, code, manager, rng):
        sd = self.util.splittable_dimensions(code)
        if len(code) <= 3 or len(sd) < 1:
            return None, code
        ud = rng.choice(sd, 1)[0]
        return self.util.split_code(code, ud, manager)

    def optimal_split(self, code, manager):
        best_cost = None
        best_code_def = None
        best_code_use = code
        best_dim = None
        if len(code) <= 3:
            return None, code
        for ud in self.util.splittable_dimensions(code):
            code_def, code_use = self.util.split_code(code, ud, None)
            cost1 = self.util.code_to_cost(code_def) + self.util.code_to_cost(code_use)
            if best_cost is None or cost1 < best_cost:
                best_cost = cost1
                best_code_def = code_def
                best_code_use = code_use
                best_dim = ud
        best_code_def, best_code_use = self.util.split_code(code, best_dim, manager)
        return best_code_def, best_code_use

    def __str__(self):
        lines = ""
        total_cst = 0
        orig_cst = self.util.code_to_cost(self.code)
        code_strs = [self.util.code_to_str(c) for c in self.codes]
        code_csts = [self.util.code_to_cost(c) for c in self.codes]
        nm = max(len(s) for s in code_strs)
        for i in range(len(code_csts)):
            total_cst += code_csts[i]
            lines = (
                lines
                + f"{code_strs[i] + ' ' * (nm + 1 - len(code_strs[i]))}   at cost = {code_csts[i]:.2e}\n  "
            )
        return f"""
 Direct contraction:

  {self.util.code_to_str(self.code)}   at cost = {orig_cst:.2e}

 Optimal contraction path:

  {lines}
 Total cost of optimal path = {total_cst:.2e} or a {orig_cst / total_cst:.2e} speedup
        """

    def commit_single_contract(self, kernel, code, bm, use_gemm):
        verbose = g.default.is_verbose("contract_plan")

        # target may need a re-shape if temporary memory was re-used
        target_shape = tuple(self.util.tags[i][1] for i in code[0][1:])
        assert int(np.prod(code[0][0].shape)) == int(np.prod(target_shape))
        if code[0][0].shape != target_shape:
            code[0][0].reshape(target_shape)

        # can I fall back on faster gemm?
        if len(code) == 3:
            C = code[0][1:]
            A = code[1][1:]
            B = code[2][1:]
            contraction_indices = (set(A) | set(B)) - set(C) - set("*")
            if isinstance(code[2][0], linear_map) and isinstance(code[1][0], g.accelerator.buffer):
                code[1], code[2] = code[2], code[1]
            if isinstance(code[1][0], linear_map):
                if code[1][0].commit_single_contract(self.util, kernel, code, bm):
                    if verbose:
                        g.message(f"Perform linear map {str(code[1][0])}")
                    return
            elif (
                len(contraction_indices) == 1
                and use_gemm
                and all(isinstance(c[0], g.accelerator.buffer) for c in code)
            ):
                i = contraction_indices.pop()
                # now test that contraction index only appears once in A and B each and split them
                if A.count(i) == 1 and B.count(i) == 1:
                    A_dag = "*" in A
                    B_dag = "*" in B
                    if A_dag:
                        A_dag, B_dag = B_dag, A_dag
                        A, B = B, A
                    A = [x for x in A if x != "*"]
                    B = [x for x in B if x != "*"]
                    A_order = [A.index(x) for x in C + [i] if x in A]
                    B_order = [B.index(x) for x in C + [i] if x in B]
                    assert len(A_order) == len(A)
                    assert len(B_order) == len(B)
                    ncommon = len(A) + len(B) - len(C) - 2
                    assert "*" not in C
                    A_reordered = [A[i] for i in A_order]
                    B_reordered = [B[i] for i in B_order]
                    AB_rest = A_reordered[0:-1] + B_reordered[ncommon:-1]
                    if AB_rest == C:
                        target = code[0][0]
                        A_reshape = tuple(self.util.tags[i][1] for i in A_reordered)
                        B_reshape = tuple(self.util.tags[i][1] for i in B_reordered)

                        A_transposed = bm.request(shape=A_reshape, dtype=target.dtype)
                        B_transposed = bm.request(shape=B_reshape, dtype=target.dtype)

                        kernel.transpose(A_transposed, code[1][0], A_order)
                        kernel.transpose(B_transposed, code[2][0], B_order)

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
                        if not A_dag and not B_dag:
                            kernel.gemm(1.0, A_transposed[idx], B_transposed[idx].T, 0.0, target[idx])
                        elif not A_dag and B_dag:
                            kernel.gemm(1.0, A_transposed[idx], B_transposed[idx].H, 0.0, target[idx])
                        else:
                            raise ValueError(f"GEMM {A_dag} - {B_dag} not yet implemented")

                        if verbose:
                            g.message(
                                f"Replace contraction by GEMM : {int(np.prod(A_reshape[0:ncommon]))} x {A_transposed.shape[ncommon:]} @ {B_transposed.shape[ncommon:]}.T"
                            )

                        # restore target shape
                        target.reshape(target_shape)

                        bm.release(A_transposed)
                        bm.release(B_transposed)
                        return

        # submit contraction if no linear operator is present
        assert all(isinstance(c[0], g.accelerator.buffer) for c in code)
        kernel.contract(*code)

    def __call__(self, kernel, optimal=True, use_gemm=True):
        if optimal:
            bm = g.accelerator.buffer_manager()

            for c in self.codes:

                tag = str([(str(x[0]), x[1:]) for x in c])
                atc = auto_tuned_class(tag, [False, True], use_gemm)
                this_use_gemm = atc.get_tuned_parameters()
                if this_use_gemm is None:
                    # do tuning

                    g.default.push_verbose("kernel", False)
                    g.default.push_verbose("contract_plan", False)

                    # Time
                    test_kernel = g.kernel()
                    self.commit_single_contract(test_kernel, c, bm, False)
                    test_kernel()  # warmup
                    t0 = -g.time()
                    test_kernel()
                    t0 += g.time()

                    test_kernel = g.kernel()
                    self.commit_single_contract(test_kernel, c, bm, True)
                    test_kernel()  # warmup
                    t1 = -g.time()
                    test_kernel()
                    t1 += g.time()

                    g.default.pop_verbose()
                    g.default.pop_verbose()

                    this_use_gemm = t1 < t0

                    atc.save_tuned_parameters(this_use_gemm)

                self.commit_single_contract(kernel, c, bm, this_use_gemm)
        else:
            # submit contraction if no linear operator is present
            assert all(isinstance(c[0], g.accelerator.buffer) for c in self.code)
            kernel.contract(*self.code)
        return kernel
