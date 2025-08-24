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
import itertools


class diagrams:
    def __init__(self, graph):
        self.graph = graph

    def __str__(self):
        s = ""
        for coef, diag in self.graph:
            s = f"{s}+ ({coef}) * {diag}\n"
        return s

    def replace(self, src, dst):
        return diagrams(
            [(a, [(dst if x == src else x, y, z) for x, y, z in b]) for a, b in self.graph]
        )

    def simplify(self):
        # for each diagram, get hash
        groups = {}
        new_graph = []
        for coef, diag in self.graph:
            fields = sorted(
                [
                    a[0]
                    + "_"
                    + (a[1] if a[1][0] != "*" else "")
                    + "_"
                    + (a[2] if a[2][0] != "*" else "")
                    for a in diag
                ]
            )
            fields_hash = "/".join(fields)
            if fields_hash not in groups:
                groups[fields_hash] = []
            groups[fields_hash].append((coef, diag))
        for group in groups:
            new_graph = new_graph + simplify_group(groups[group])
        return diagrams(new_graph)

    def topologies(self):
        return [match_string(diag) for coef, diag in self.graph]

    def coefficients(self, names):
        coefs = {}
        for coef, diag in self.graph:
            ms = match_string(diag)
            if ms in names:
                name = names[ms]
            else:
                name = ms
            if name in coefs:
                coefs[name] += coef
            else:
                coefs[name] = coef
        return coefs


def match_string(diag):
    s = []
    for a, b, c in diag:
        if a[0].isupper():
            args = sorted([b, c])
            s.append(f"{a}_{args[0]}_{args[1]}")
        else:
            s.append(f"{a}_{b}_{c}")
    return "/".join(sorted(s))


def match(diag0, diag1):
    diag0_ms = match_string(diag0)
    # need list of internal coordinates
    ic = list(
        set([x[1] for x in diag1 if x[1][0] == "*"] + [x[2] for x in diag1 if x[2][0] == "*"])
    )
    xi = list(
        set([x[1] for x in diag1 if x[1][0] != "*"] + [x[2] for x in diag1 if x[2][0] != "*"])
    )
    # try matching for each permutation of internal coordinates
    for pic in itertools.permutations(ic):
        # permute indices
        index_map = {ic[i]: pic[i] for i in range(len(ic))}
        for i in range(len(xi)):
            index_map[xi[i]] = xi[i]
        diag1_prime = [(a, index_map[b], index_map[c]) for a, b, c in diag1]
        diag1_ms = match_string(diag1_prime)
        if diag0_ms == diag1_ms:
            return True
    return False


def simplify_group(group):
    new_group = []
    for coef, diag in group:
        # is already in group
        add = True
        for i in range(len(new_group)):
            if match(new_group[i][1], diag):
                new_group[i][0] += coef
                add = False
                break
        if add:
            new_group.append([coef, diag])
    # remove if zero coefficient
    return [(coef, diag) for coef, diag in new_group if abs(coef) > 1e-13]
