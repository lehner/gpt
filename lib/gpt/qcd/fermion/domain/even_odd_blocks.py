#
#    GPT - Grid Python Toolkit
#    Copyright (C) 2020  Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)
#                  2020  Mattia Bruno
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
from gpt.params import params_convention


def set_domain_boundaries_of_U(U, domain):
    colon = slice(None, None, None)
    nd = len(U)
    for dim in range(nd):
        for i in range(1, domain.extended_local_blocks_per_dimension[dim] + 1):
            U[dim][
                tuple([colon] * dim + [i * domain.block_size[dim] - 1] + [colon] * (nd - dim - 1))
            ] = 0


def domain_fermion_operator(op, domain):
    U_domain = []
    for u in op.U:
        u_domain = domain.lattice(u.otype)
        domain.project(u_domain, u)
        U_domain.append(u_domain)
    set_domain_boundaries_of_U(U_domain, domain)
    return op.updated(U_domain)


class even_odd_blocks:
    @params_convention(block_size=None)
    def __init__(self, fermion, params):
        U_bs = params["block_size"]
        assert U_bs is not None

        # for now ad-hoc treatment of 5d fermions
        if fermion.F_grid.nd == len(U_bs) + 1:
            F_bs = [fermion.F_grid.fdimensions[0]] + U_bs
        else:
            F_bs = U_bs

        # create domains
        self.F_domains = [
            gpt.domain.even_odd_blocks(fermion.F_grid, F_bs, gpt.even),
            gpt.domain.even_odd_blocks(fermion.F_grid, F_bs, gpt.odd),
        ]

        self.U_domains = [
            gpt.domain.even_odd_blocks(fermion.U_grid, U_bs, gpt.even),
            gpt.domain.even_odd_blocks(fermion.U_grid, U_bs, gpt.odd),
        ]

        # create operators living on domains
        self.fermions = [domain_fermion_operator(fermion, domain) for domain in self.U_domains]

        self.F_bs = F_bs
        self.U_bs = U_bs
