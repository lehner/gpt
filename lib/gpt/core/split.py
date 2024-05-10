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
import cgpt
import numpy
import sys


# Implement policies as classes, may want to add more variables/methods later
class split_group_policy:
    class together:
        pass

    class separate:
        pass


def split_lattices(lattices, lcoor, gcoor, split_grid, N, cache, group_policy):
    # Example:
    #
    # Original
    #
    # lattice1,...,latticen | lattice1,...,latticen
    #
    # New
    #
    # lattice1,...,latticeN | latticeN+1,...,lattice2N
    #
    # Q = n // N = 2

    # N is desired number of parallel split lattices per unsplit lattice
    # 1 <= N <= sranks, sranks % N == 0

    n = len(lattices)
    assert n > 0
    assert n % N == 0
    Q = n // N

    # Save memory by performing each group separately
    if N != 1 and group_policy == split_group_policy.separate:
        res = []
        for i in range(N):
            res += split_lattices(
                [lattices[q * N + i] for q in range(Q)],
                lcoor,
                gcoor,
                split_grid,
                1,
                cache,
                group_policy,
            )
        return res

    assert len(lcoor) == len(gcoor)
    grid = lattices[0].grid
    assert all([lattices[i].grid.obj == grid.obj for i in range(1, n)])
    cb = lattices[0].checkerboard()
    assert all([lattices[i].checkerboard() is cb for i in range(1, n)])
    otype = lattices[0].otype
    assert all([lattices[i].otype.__name__ == otype.__name__ for i in range(1, n)])

    l = [gpt.lattice(split_grid, otype) for i in range(N)]

    for x in l:
        x.checkerboard(cb)
        x.split_lcoor = lcoor
        x.split_gcoor = gcoor
    sranks = split_grid.sranks
    srank = split_grid.srank

    src_data = lattices
    dst_data = l

    # build views
    if cache is None:
        cache = {}

    cache_key = f"split_plan_{lattices[0].grid.obj}_{l[0].grid.obj}_{lattices[0].otype.__name__}_{l[0].otype.__name__}_{n}_{N}"
    if cache_key not in cache:
        plan = gpt.copy_plan(dst_data, src_data, embed_in_communicator=lattices[0].grid)
        i = srank // (sranks // Q)
        for x in lattices[i * N : (i + 1) * N]:
            plan.source += x.view[gcoor]
        for x in l:
            plan.destination += x.view[lcoor]
        cache[cache_key] = plan()

    cache[cache_key](dst_data, src_data)

    return l


def unsplit(first, second, cache=None, group_policy=split_group_policy.separate):
    if not isinstance(first, list):
        return unsplit([first], [second])

    n = len(first)
    N = len(second)
    Q = n // N
    assert n % N == 0

    # Save memory by performing each group separately
    if N != 1 and group_policy == split_group_policy.separate:
        for i in range(N):
            unsplit([first[q * N + i] for q in range(Q)], [second[i]], cache, group_policy)
        return

    split_grid = second[0].grid
    sranks = split_grid.sranks
    srank = split_grid.srank

    lcoor = second[0].split_lcoor
    gcoor = second[0].split_gcoor

    src_data = second
    dst_data = first

    if cache is None:
        cache = {}

    cache_key = f"unsplit_plan_{first[0].grid.obj}_{second[0].grid.obj}_{first[0].otype.__name__}_{second[0].otype.__name__}_{n}_{N}"
    if cache_key not in cache:
        plan = gpt.copy_plan(dst_data, src_data, embed_in_communicator=first[0].grid)
        i = srank // (sranks // Q)
        for x in first[i * N : (i + 1) * N]:
            plan.destination += x.view[gcoor]
        for x in second:
            plan.source += x.view[lcoor]
        cache[cache_key] = plan()

    cache[cache_key](dst_data, src_data)


def split_by_rank(first, group_policy=split_group_policy.separate):
    if not isinstance(first, list):
        return split_by_rank([first])[0]

    assert len(first) > 0

    # TODO: split types
    lattices = first
    grid = lattices[0].grid
    mpi_split = [1, 1, 1, 1]
    fdimensions = [grid.fdimensions[i] // grid.mpi[i] for i in range(grid.nd)]
    split_grid = grid.split(mpi_split, fdimensions)
    gcoor = gpt.coordinates(lattices[0])
    lcoor = gpt.coordinates((split_grid, lattices[0].checkerboard()))
    return split_lattices(lattices, lcoor, gcoor, split_grid, len(lattices), group_policy)


def split(first, split_grid, cache=None, group_policy=split_group_policy.separate):
    assert len(first) > 0
    lattices = first
    gcoor = gpt.coordinates((split_grid, lattices[0].checkerboard()))
    lcoor = gpt.coordinates((split_grid, lattices[0].checkerboard()))
    assert len(lattices) % split_grid.sranks == 0
    return split_lattices(
        lattices,
        lcoor,
        gcoor,
        split_grid,
        len(lattices) // split_grid.sranks,
        cache,
        group_policy,
    )


# class split_map:
#     def __init__(self, mpi_split):
#         self.mpi_split = mpi_split
#         self.split_grids = {}
#         self.verbose = gpt.default.is_verbose("split_map")

#     def __call__(self, functions, outputs, inputs = None, groups = None):

#         n_functions = len(functions)
#         n_outputs = len(outputs)

#         assert n_functions == n_outputs

#         if groups is None:
#             groups = [[i] for i in range(n_functions)]

#         one_argument_call = inputs is None
#         if inputs is None:
#             inputs = [[] for i in range(n_outputs)]

#         n_inputs = len(inputs)

#         assert n_inputs == n_outputs

#         n_groups = len(groups)

#         if self.verbose:
#             gpt.message(f"Split map for {n_functions} functions in {n_groups} groups")

#         all_grids = [x.grid for y in outputs + inputs for x in y]
#         grids = []
#         for gr in all_grids:
#             if gr not in grids:
#                 grids.append(gr)

#         srank = None
#         sranks = None
#         log_node = True

#         # split grids
#         for gr in grids:
#             if gr not in self.split_grids:
#                 mpi_split = [1]*(gr.nd - len(self.mpi_split)) + self.mpi_split
#                 self.split_grids[gr] = (gr.split(mpi_split, gr.fdimensions),{})

#             sgr = self.split_grids[gr][0]
#             if srank is not None:
#                 assert sgr.srank == srank and sgr.sranks == sranks
#             else:
#                 srank, sranks = sgr.srank, sgr.sranks
#                 log_node = sgr.processor == 0

#         # my jobs
#         srank_jobs = [[j for gr in groups[s::sranks] for j in gr] for s in range(sranks)]

#         if self.verbose:
#             gpt.message(f"S[{srank}/{sranks}] : jobs {srank_jobs[srank]}", force_output = log_node)

#         # go through each srank and make sure they have the fields they need
#         t = gpt.timer("split map call")

#         t("field layout")
#         this_srank_fields = {}
#         for i in range(sranks):
#             for j in srank_jobs[i]:
#                 for f in inputs[j] + outputs[j]:
#                     sgrid, cache = self.split_grids[f.grid]

#                     # create a target
#                     target = gpt.lattice(sgrid, f.otype)
#                     target.checkerboard(f.checkerboard())

#                     ckey = target.checkerboard()
#                     if ckey not in cache:
#                         cache[ckey] = gpt.coordinates(target)

#                     # fill it
#                     cc = cache[ckey] if i == srank else cache[ckey][0:0]

#                     dkey = f"{ckey.__name__}_{f.otype.__name__}_{i}"
#                     if dkey not in cache:
#                         cache[dkey] = {}

#                     target[cc, cache[dkey]] = f[cc, cache[dkey]]

#                     # keep it
#                     if i == srank:
#                         if j not in this_srank_fields:
#                             this_srank_fields[j] = []
#                         this_srank_fields[j].append(target)

#         t("wait")

#         all_grids[0].barrier()

#         t("compute")

#         # now perform my jobs
#         results = numpy.zeros(shape=(n_functions,), dtype=numpy.complex128)
#         for j in srank_jobs[srank]:

#             check = list(set([(x.grid.srank,x.grid.sranks) for x in this_srank_fields[j]] + [(srank,sranks)]))
#             assert len(check) == 1

#             if one_argument_call:
#                 r = functions[j](this_srank_fields[j])
#             else:
#                 r = functions[j](this_srank_fields[j][len(inputs[j]):], this_srank_fields[j][0:len(inputs[j])])
#             if gpt.util.is_num(r) and log_node:
#                 results[j] = r

#         t("wait")

#         all_grids[0].barrier()

#         sys.exit(0)

#         t("field layout")

#         for i in range(sranks):
#             for j in srank_jobs[i]:
#                 for idx, f in enumerate(outputs[j]):
#                     sgrid, cache = self.split_grids[f.grid]

#                     if j in this_srank_fields:
#                         target = this_srank_fields[j][len(inputs[j]) + idx]
#                     else:
#                         target = gpt.lattice(sgrid, f.otype)
#                         target.checkerboard(f.checkerboard())

#                     # coordinates
#                     ckey = target.checkerboard()
#                     cc = cache[ckey] if i == srank else cache[ckey][0:0]

#                     dkey = f"{ckey.__name__}_{f.otype.__name__}_{i}_r"
#                     if dkey not in cache:
#                         cache[dkey] = {}

#                     f[cc, cache[dkey]] = target[cc, cache[dkey]]

#         t()

#         all_grids[0].globalsum(results)

#         if self.verbose:
#             gpt.message(f"S[{srank}/{sranks}] : {t}", force_output = log_node)

#         return results
