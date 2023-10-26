#
#    GPT - Grid Python Toolkit
#    Copyright (C) 2023  Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)
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


def einsum(contraction, *tensors):
    contraction = contraction.split("->")
    if len(contraction) != 2:
        raise Exception(f"{contraction} needs to be explicit, i.e., of the form ...->...")
    source, destination = contraction
    source = [[x for x in s] for s in source.split(",")]
    destination = [[x for x in s] for s in destination.split(",")]
    if len(tensors) != len(source) + len(destination):
        raise Exception(f"Need {len(source)} source and {len(destination)} destination tensors")
    tensors_source = tensors[0 : len(source)]
    tensors_destination = tensors[len(source) :]

    # now infer and verify index dimensions
    index_dimension = {}
    epsilon_indices = {}
    epsilon_tensors = []
    source_indices = {}
    destination_indices = {}
    for indices, tensors, all_indices in [
        (source, tensors_source, source_indices),
        (destination, tensors_destination, destination_indices),
    ]:
        for i in range(len(indices)):
            if tensors[i] is g.epsilon:
                dim = len(indices[i])
                epsilon_tensors.append(indices[i])
                for s in indices[i]:
                    all_indices[s] = True
                    epsilon_indices[s] = True
                    if s in index_dimension:
                        if index_dimension[s] != dim:
                            raise Exception(f"Index {s} already defined to have dimension {dim}")
                    else:
                        index_dimension[s] = dim
            else:
                shape = tensors[i].otype.shape
                if shape == (1,):
                    shape = tuple()
                if len(shape) != len(indices[i]):
                    raise Exception(
                        f"Tensor {i} is expected to have {len(shape)} indices instead of {len(indices[i])}"
                    )
                for j in range(len(shape)):
                    dim = shape[j]
                    s = indices[i][j]
                    all_indices[s] = True
                    if s in index_dimension:
                        if index_dimension[s] != dim:
                            raise Exception(f"Index {s} already defined to have dimension {dim}")
                    else:
                        index_dimension[s] = dim
    # print(index_dimension)
    # now go through all indices
    indices = list(index_dimension.keys())
    full_indices = [i for i in destination_indices if i not in epsilon_indices]
    nsegment = 1
    for i in full_indices:
        nsegment *= index_dimension[i]
    for i in source_indices:
        if i not in epsilon_indices and i not in full_indices:
            full_indices.append(i)
    index_value = [0] * len(full_indices)

    code = []

    def get_element(indices, names, values):
        element = 0
        for i in indices:
            element = element * index_dimension[i] + values[names.index(i)]
        return element

    acc = {}
    ti = g.stencil.tensor_instructions

    def process(names, values, sign):
        # for now can only do single destination tensor and two source tensor lattices
        assert len(destination) == 1
        c = destination[0]
        sidx = []
        for i in range(len(source)):
            if tensors_source[i] is not g.epsilon:
                sidx.append(source[i])

        # get destination index
        c_element = get_element(c, names, values)

        if len(sidx) == 2:
            a_element = get_element(sidx[0], names, values)
            b_element = get_element(sidx[1], names, values)

            if c_element not in acc:
                acc[c_element] = True
                mode = ti.mov if sign > 0 else ti.mov_neg
            else:
                mode = ti.inc if sign > 0 else ti.dec
            code.append((0, c_element, mode, 1.0, [(1, 0, a_element), (2, 0, b_element)]))

        elif len(sidx) == 1:
            a_element = get_element(sidx[0], names, values)
            if c_element not in acc:
                acc[c_element] = True
                mode = ti.mov if sign > 0 else ti.mov_neg
            else:
                mode = ti.inc if sign > 0 else ti.dec
            code.append((0, c_element, mode, 1.0, [(1, 0, a_element)]))

        else:
            raise Exception(
                "General einsum case not yet implemented; limited to contraction of one or two tensors"
            )

    def process_indices(names, values, epsilon_tensors, sign0):
        if len(epsilon_tensors) == 0:
            process(names, values, sign0)
        else:
            n = len(epsilon_tensors[0])
            eps = g.epsilon(n)
            for i, sign in eps:
                keep = True
                for j in range(n):
                    idx = epsilon_tensors[0][j]
                    if idx in names and values[names.index(idx)] != i[j]:
                        keep = False
                        break
                if keep:
                    names_next = [n for n in names]
                    values_next = [v for v in values]
                    for j in range(n):
                        idx = epsilon_tensors[0][j]
                        if idx not in names:
                            names_next.append(idx)
                            values_next.append(i[j])
                    process_indices(names_next, values_next, epsilon_tensors[1:], sign * sign0)

    active = True
    while active:
        process_indices(full_indices, index_value, epsilon_tensors, 1)
        for j in range(len(index_value)):
            if index_value[j] + 1 < index_dimension[full_indices[j]]:
                index_value[j] += 1
                break
            elif j == len(index_value) - 1:
                active = False
            else:
                index_value[j] = 0

    assert len(code) % nsegment == 0
    segments = [(len(code) // nsegment, nsegment)]

    ein = g.stencil.tensor(tensors_destination[0], [(0, 0, 0, 0)], code, segments)

    def exec(*src):
        c = g.lattice(tensors_destination[0])
        ein(c, *src)
        return c

    return exec
