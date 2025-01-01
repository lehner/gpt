#
#    GPT - Grid Python Toolkit
#    Copyright (C) 2024-25  Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)
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
import numpy as np
import gpt as g


def sparse_domain_conformable(a_sdomain, b_sdomain):
    a = a_sdomain.coordinate_lattices()
    b = b_sdomain.coordinate_lattices()
    if len(a) != len(b):
        return False
    for mu in range(len(a)):
        eps2 = g.norm2(g.convert(b[mu], g.double) - g.convert(a[mu], g.double))
        if eps2 > 1e-13:
            return False
    return True


class source_domain:
    def __init__(self, sampled_sites):
        self.sampled_sites = sampled_sites

    def __eq__(self, other):
        return self.sampled_sites == other.sampled_sites

    def restrict(self, other):
        if other.sampled_sites < self.sampled_sites:
            self.sampled_sites = other.sampled_sites


class sink_domain:
    def __init__(self, header):
        self.sdomain = header["sparse_domain"]
        self.sampled_sites = header["number_of_sink_positions"]
        self.L = np.array(self.sdomain.grid.gdimensions, dtype=np.int32)
        self.total_sites = float(np.prod(self.L.astype(np.float64)))
        self.coordinates = header["all_positions"]

    def __eq__(self, other):
        return sparse_domain_conformable(self.sdomain, other.sdomain)


class flavor_base:
    def __init__(self, source_domain, sink_domain, filename):
        self.source_domain = source_domain
        self.sink_domain = sink_domain
        self.filename = filename


class flavor_multi:
    def __init__(self, array, cache_size, cache_line_size):
        self.array = array

        assert cache_line_size <= cache_size
        assert cache_size % cache_line_size == 0

        # test conformality
        flav0 = array[0][1]
        self.source_domain = flav0.source_domain
        self.sink_domain = flav0.sink_domain

        max_source_sampled_sites = max([flav.source_domain.sampled_sites for fac, flav in array])

        for fac, flav in array[1:]:
            assert flav0.sink_domain == flav.sink_domain
            self.source_domain.restrict(flav.source_domain)

        self.coordinates = self.sink_domain.coordinates

        self.ec = self.sink_domain.sdomain.unique_embedded_coordinates(
            self.coordinates[0:max_source_sampled_sites]
        )

        self.cache_size = cache_size
        self.cache_line_size = cache_line_size
        self.cache = []
        self.cache_hits = 0
        self.cache_misses = 0

    def get_propagator_full(self, i):

        # find my cache line index
        cache_line_idx = i // self.cache_line_size
        cache_line_offset = cache_line_idx * self.cache_line_size
        idx_within_cache_line = i - cache_line_offset

        # find in cache
        for idx in reversed(range(len(self.cache))):
            if self.cache[idx][0] == cache_line_idx:
                cc = self.cache.pop(idx)
                self.cache.append(cc)
                self.cache_hits += 1
                return cc[1][idx_within_cache_line]

        # if not in cache, create
        i0 = cache_line_offset
        i1 = min(i0 + self.cache_line_size, self.source_domain.sampled_sites)
        ilist = list(range(i0, i1))

        self.cache_misses += len(ilist)

        # make room in cache
        cache_length = self.cache_size // self.cache_line_size
        while len(self.cache) >= cache_length:
            self.cache.pop(0)

        g.message(
            f"Create cache entry for [{i0},...,{i1 - 1}]; statistics: hits = {self.cache_hits}, misses = {self.cache_misses}"
        )

        # load cache line
        keys = [f"{self.coordinates[il].tolist()}" for il in ilist]
        paths = [f"/{key}/propagator" for key in keys]
        g.default.push_verbose("io", False)
        data = {flav: g.load(flav.filename, paths=paths) for fac, flav in self.array}
        g.default.pop_verbose()

        # process cache
        prp = []
        for il, key in zip(ilist, keys):
            prp_il = None
            for fac, flav in self.array:
                prop = g.convert(data[flav][1 + il][key]["propagator"], g.double)
                if prp_il is None:
                    prp_il = g(fac * prop)
                else:
                    prp_il += fac * prop
            prp.append(prp_il)

        self.cache.append((cache_line_idx, prp))

        return prp[idx_within_cache_line]

    def __getitem__(self, args):

        if isinstance(args, tuple):
            i, without = args
        else:
            i, without = args, None

        prp = self.get_propagator_full(i)

        # sparsen sink if requested
        if without is not None:
            prp = g.copy(prp)
            without = np.ascontiguousarray(self.ec[without])
            prp[without] = 0

        return prp

    def __call__(self, i_sink, i_src):
        # should also work if I give a list of i_sink
        prp = self.get_propagator_full(i_src)
        return prp[self.ec[i_sink]]

    def source_mask(self):
        mask = self.sink_domain.sdomain.lattice(g.ot_complex_additive_group())
        mask[:] = 0
        mask[self.ec] = 1
        return mask


def cache_optimized_sampler(flavors, original):
    remainder = [x for x in original]

    cache_checks = 0
    dt = -g.time()

    while len(remainder) > 0:
        cached = [(fl.cache_line_size, set([x[0] for x in fl.cache])) for fl in flavors]

        # find all tuples which are cached right now
        for i, element in enumerate(remainder):
            cache_checks += 1
            if all([e // c[0] in c[1] for e, c in zip(element, cached)]):
                remainder.pop(i)
                dt += g.time()
                yield element
                dt -= g.time()

        if len(remainder) > 0:
            # any element will do next
            dt += g.time()
            yield remainder.pop(0)
            dt -= g.time()

    dt += g.time()
    g.message(f"Optimized sampler needed {cache_checks} cache checks; overhead of {dt} seconds")


#
# Load quarks
#
cache = {}


def get_quark(fn):
    global cache
    if fn not in cache:
        g.message(f"Load {fn}")
        quark = g.load(fn, paths=["/header/*"])

        # sources
        number_of_sources = len(quark) - 1
        src_domain = source_domain(number_of_sources)

        # embed them in double precision
        sdomain = quark[0]["header"]["sparse_domain"]
        grid = sdomain.kernel.grid
        if grid.precision is not g.double:
            g.message(f"Embed domain {fn} in double precision fields")

            sdomain_cl = sdomain.coordinate_lattices()

            mask = (sdomain_cl[0][:] >= 0)[:, 0]
            local_coordinates = np.hstack(tuple([x[:].real.astype(np.int32) for x in sdomain_cl]))

            grid_dp = grid.converted(g.double)

            sdomain_dp = g.domain.sparse(
                grid_dp,
                local_coordinates,
                dimensions_divisible_by=sdomain_cl[0].grid.fdimensions,
                mask=mask,
            )

            assert sparse_domain_conformable(sdomain, sdomain_dp)
            quark[0]["header"]["sparse_domain"] = sdomain_dp
            g.message("Done")

        # sink
        snk_domain = sink_domain(quark[0]["header"])

        # flavor
        cache[fn] = flavor_base(src_domain, snk_domain, fn)

    return cache[fn]


prop = {}


def flavor(root, *cache_param):
    global prop

    # TODO: if root is a list, return a list of flavors that have
    # a uniform sink sparse domain
    tag, prec = root.split(".")

    ttag = f"{tag}.{prec}"
    if ttag in prop:
        return prop[ttag]

    sloppy_file = f"{tag}/full/sloppy"
    exact_file = f"{tag}/full/exact"

    if prec == "s":
        prop[ttag] = flavor_multi([(1.0, get_quark(sloppy_file))], *cache_param)
    elif prec == "e":
        prop[ttag] = flavor_multi([(1.0, get_quark(exact_file))], *cache_param)
    elif prec == "ems":
        prop[ttag] = flavor_multi(
            [(1.0, get_quark(exact_file)), (-1.0, get_quark(sloppy_file))], *cache_param
        )
    else:
        raise Exception(f"Unknown precision: {prec}")

    return prop[ttag]
