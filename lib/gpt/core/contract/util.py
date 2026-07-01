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
from gpt.core.contract.linear_map import linear_map


class util:
    def __init__(self, code):
        assert all(isinstance(x, (list, tuple)) for x in code)
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

    def shape_from_dimensions(self, dimensions):
        return tuple(self.tags[x][1] for x in dimensions)

    def participants_witnesses(self, code, d):
        # re-order participants such that those come first with lower indices w.r.t. initial contraction order
        sort_basis = len(self.sorted_dimensions)
        participants = sorted(
            [c for c in code[1:] if d in c[1:]],
            key=lambda x: sum(
                sort_basis**-i * self.sorted_dimensions.index(y)
                for i, y in enumerate(x[1:])
                if y != "*"
            ),
        )

        # other factors that are not part of split are called witnesses
        witnesses = [c for c in code[1:] if d not in c[1:]]
        return participants, witnesses

    def participants_dimensions(self, participants, d):
        # get a list of all used dimensions in factors that we need for split
        ud = self.used_dimensions(participants)

        # remove CC and split dimension; this keeps genuine external indices to split
        ud = [x for x in ud if x not in [d, "*"]]

        # create an ordered list of indices for definition of new tensor; prioritize common indices, then in order of participants
        ud_ordered = [x for x in ud if all(x in y[1:] for y in participants)]
        for c in participants:
            ud_new = list((set(c[1:]) & set(ud)) - set(ud_ordered))
            ud_ordered.extend([x for x in ud if x in ud_new])

        return ud_ordered

    def split_code(self, code, d, manager):
        # cannot split over an index that needs to go into the target
        assert d not in code[0][1:]

        # create participants / witness list
        participants, witnesses = self.participants_witnesses(code, d)

        # get participants tensor definition
        ud_ordered = self.participants_dimensions(participants, d)

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

    def is_splittable(self, code, c):
        p, w = self.participants_witnesses(code, c)
        if any(isinstance(x[0], linear_map) for x in p):
            # need one linear_map and one accelerator_buffer
            if len(p) == 2 and any(isinstance(x[0], g.core.accelerator_buffer) for x in p):
                return True
            return False
        return True

    def splittable_dimensions(self, code):
        ud = self.used_dimensions(code)
        target_indices = code[0][1:]
        return [x for x in ud if x not in target_indices and self.is_splittable(code, x)]

    def code_to_cost(self, code):
        return np.prod([self.tags[x][1] for x in self.used_dimensions(code)])

    def optimize_general(self, code, temporary_manager, split):
        code_use = code
        codes = []
        temporaries = set([])
        while True:
            code_def, code_use = split(code_use, temporary_manager)
            if code_def is None:
                break
            temporaries.add(code_def[0][0])
            codes.append(code_def)

            # see if I can remove temporaries
            still_used = set(x[0] for x in code_use if x[0] in temporaries)
            removable = temporaries - still_used
            if temporary_manager is not None:
                for t in removable:
                    temporary_manager.release(t)
                    temporaries.remove(t)

        codes.append(code_use)

        if temporary_manager is not None:
            for t in temporaries:
                temporary_manager.release(t)
        return codes
