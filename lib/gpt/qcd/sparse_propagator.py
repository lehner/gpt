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


class source_domain:
    def __init__(self, sampled_sites):
        self.sampled_sites = sampled_sites

    def copy(self):
        return source_domain(self.sampled_sites)

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
        self.coordinates = header["all_positions"][0 : self.sampled_sites]

    def copy(self):
        return sink_domain(
            {
                "sparse_domain": self.sdomain,
                "number_of_sink_positions": self.sampled_sites,
                "all_positions": self.coordinates,
            }
        )

    def __eq__(self, other):
        return self.sdomain.conformable(other.sdomain)

    def restrict(self, other):
        if other.sampled_sites < self.sampled_sites:
            self.sampled_sites = other.sampled_sites
            self.sdomain = other.sdomain
            self.coordinates = self.coordinates[0 : self.sampled_sites]
            # sdomain is always smallest domain


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
        self.source_domain = flav0.source_domain.copy()
        self.sink_domain = flav0.sink_domain.copy()

        for tag, flav in array[1:]:
            self.sink_domain.restrict(flav.sink_domain)
            self.source_domain.restrict(flav.source_domain)

        self.cache_size = cache_size
        self.cache_line_size = cache_line_size
        self.cache = []
        self.cache_hits = 0
        self.cache_misses = 0

    def sink_domain_update(self, min_source_sampled_sites):
        self.cache = []
        self.coordinates = self.sink_domain.coordinates
        self.ec = self.sink_domain.sdomain.unique_embedded_coordinates(
            self.coordinates[0:min_source_sampled_sites]
        )

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
        data = {flav: g.load(flav.filename, paths=paths) for tag, flav in self.array}
        g.default.pop_verbose()

        # process cache
        prp = []
        for il, key in zip(ilist, keys):
            prp_il = {}
            for tag, flav in self.array:
                prp_il[tag] = g.convert(data[flav][1 + il][key]["propagator"], g.double)
                if self.sink_domain.sdomain is not flav.sink_domain.sdomain:
                    prp_il[tag] = g(
                        self.sink_domain.sdomain.project
                        * flav.sink_domain.sdomain.promote
                        * prp_il[tag]
                    )
            prp.append(prp_il)

        self.cache.append((cache_line_idx, prp))

        return prp[idx_within_cache_line]

    def __getitem__(self, args):

        assert isinstance(args, tuple)
        if len(args) == 3:
            tag, i, without = args
        else:
            tag, i, without = *args, None

        prp = self.get_propagator_full(i)[tag]

        # sparsen sink if requested
        if without is not None:
            prp = g.copy(prp)
            without = np.ascontiguousarray(self.ec[without])
            prp[without] = 0

        return prp

    def __call__(self, tag, i_sink, i_src):
        # should also work if I give a list of i_sink
        prp = self.get_propagator_full(i_src)[tag]
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
            sdomain_dp = sdomain.converted(g.double)

            assert sdomain.conformable(sdomain_dp)
            quark[0]["header"]["sparse_domain"] = sdomain_dp
            g.message("Done")

        # sink
        snk_domain = sink_domain(quark[0]["header"])

        # flavor
        cache[fn] = flavor_base(src_domain, snk_domain, fn)

    return cache[fn]


prop = {}


def flavor(roots, *cache_param):
    global prop

    if not isinstance(roots, list):
        roots = [roots]

    tag_prec = [root.rsplit(".", maxsplit=1) for root in roots]
    ttag = [f"{tag}.{prec}" for tag, prec in tag_prec]
    ttag = str(ttag)

    if ttag in prop:
        return prop[ttag]

    weights = [(1.0, [])]
    ret = []
    for tag, prec in tag_prec:
        low_file = f"{tag}/full/low"
        sloppy_file = f"{tag}/full/sloppy"
        exact_file = f"{tag}/full/exact"

        if len(prec) == 1:
            for w in weights:
                w[1].append(prec)
        elif len(prec) == 3:
            weights_a = []
            weights_b = []
            for w in weights:
                weights_a.append((w[0], w[1] + [prec[0]]))
                weights_b.append((-w[0], w[1] + [prec[2]]))
            weights = weights_a + weights_b

        if prec == "s":
            prp = flavor_multi([("s", get_quark(sloppy_file))], *cache_param)
        elif prec == "e":
            prp = flavor_multi([("e", get_quark(exact_file))], *cache_param)
        elif prec == "l":
            prp = flavor_multi([("l", get_quark(low_file))], *cache_param)
        elif prec == "ems":
            prp = flavor_multi(
                [("e", get_quark(exact_file)), ("s", get_quark(sloppy_file))], *cache_param
            )
        elif prec == "sml":
            prp = flavor_multi(
                [("s", get_quark(sloppy_file)), ("l", get_quark(low_file))], *cache_param
            )
        else:
            raise Exception(f"Unknown precision: {prec}")

        ret.append(prp)

    # select common sink domain
    common_sink_domain = ret[0].sink_domain
    for flav in ret[1:]:
        common_sink_domain.restrict(flav.sink_domain)
    max_ec_size = max(
        [
            min([flav.source_domain.sampled_sites for fac, flav in flav_mult.array])
            for flav_mult in ret
        ]
    )

    # prepare common sink domain
    for flav in ret:
        flav.sink_domain = common_sink_domain
        flav.sink_domain_update(max_ec_size)

    prop[ttag] = [weights] + ret
    return prop[ttag]
