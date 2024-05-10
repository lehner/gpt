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
import itertools as it
import copy


class evaluation_context:
    def __init__(self):
        self.values = {}

    def __setitem__(self, item, value):
        self.values[item] = value

    def __getitem__(self, item):
        return self.values[item]

    def __contains__(self, item):
        return item in self.values


class fields_context:
    def __init__(self, fields=None, coordinate_arguments=None, index_arguments=None):
        if fields is None:
            fields = [{}]
        if index_arguments is None:
            index_arguments = {}
        if coordinate_arguments is None:
            coordinate_arguments = {}
        self.fields = fields
        self.coordinate_arguments = coordinate_arguments
        self.index_arguments = index_arguments

    def clone(self):
        return fields_context(
            copy.deepcopy(self.fields),
            copy.deepcopy(self.coordinate_arguments),
            copy.deepcopy(self.index_arguments),
        )

    def merge(self, other):
        self.fields = self.fields + other.fields
        for a in other.index_arguments:
            self.index_arguments[a] = other.index_arguments[a]
        for a in other.coordinate_arguments:
            self.coordinate_arguments[a] = other.coordinate_arguments[a]

    def register_field(self, index, is_bar, path, coordinate_argument, index_argument):
        self.coordinate_arguments[path] = coordinate_argument
        self.index_arguments[path] = index_argument

        for fields in self.fields:
            if index not in fields:
                fields[index] = ([], [])

            if "*" not in fields:
                fields["*"] = []

            if is_bar:
                fields[index][1].append(path)
            else:
                fields[index][0].append(path)

            fields["*"].append(path)

    def contract(self, verbose):
        result = []

        index = 0
        tag_index = {}
        index_tag = {}

        for cfields in self.fields:
            contractions = [[]]
            for f in cfields:
                if f == "*":
                    continue

                n_fermions = len(cfields[f][0])
                n_bar_fermions = len(cfields[f][1])
                if verbose:
                    g.message(
                        f"flavor {f} has {n_fermions} fields and {n_bar_fermions} matching bar fields"
                    )

                fermion_indices = []
                bar_fermion_indices = []

                if n_fermions == n_bar_fermions:
                    for x in cfields[f][0]:
                        tag_index[x] = index
                        index_tag[index] = x
                        fermion_indices.append(index)
                        index += 1

                    for x in cfields[f][1]:
                        tag_index[x] = index
                        index_tag[index] = x
                        bar_fermion_indices.append(index)
                        index += 1

                flavor_contractions = []
                for bar_permutation in it.permutations(bar_fermion_indices):
                    flavor_contractions.append(
                        [(fermion_indices[i], bar_permutation[i]) for i in range(n_fermions)]
                    )
                contractions = [c + d for c in contractions for d in flavor_contractions]

            # sign of contractions
            if "*" in cfields:
                actual_index_order = [tag_index[x] for x in cfields["*"]]
            else:
                actual_index_order = []

            for c in contractions:
                desired_index_order = []
                for a, b in c:
                    desired_index_order = desired_index_order + [a, b]

                sign = g.sign_of_permutation(actual_index_order, desired_index_order)

                tags = []
                for a, b in c:
                    tags.append((index_tag[a], index_tag[b]))

                result.append((sign, tags))

        return result
