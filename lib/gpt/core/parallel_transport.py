#
#    GPT - Grid Python Toolkit
#    Copyright (C) 2020  Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)
#
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

# import cProfile as prof


class path:
    def __init__(self, path=None):
        if path is None:
            path = []
        self.path = path

    def forward(self, mu, distance=1):
        self.path.append((mu, distance))
        return self

    def backward(self, mu, distance=1):
        self.forward(mu, -distance)
        return self

    def inverse(self):
        return path([(mu, -distance) for (mu, distance) in reversed(self.path)])


# define short-cuts
path.f = path.forward
path.b = path.backward


class parallel_transport:
    def __init__(self, links, paths, site_fields=None):
        self.paths = paths
        self.dim = len(links)

        if site_fields is None:
            site_fields = []

        self.n_site_fields = len(site_fields)

        link_displacements = [set() for mu in range(self.dim)]
        site_displacements = set()
        for p in paths:
            d = [0] * self.dim
            for mu, distance in p.path:
                assert mu >= 0 and mu < self.dim
                for step in range(abs(distance)):
                    if distance > 0:
                        link_displacements[mu].add(tuple(d))
                    d[mu] += distance // abs(distance)
                    if distance < 0:
                        link_displacements[mu].add(tuple(d))
            site_displacements.add(tuple(d))

        plan = g.cshift_plan()

        self.link_indices = []
        for mu in range(self.dim):
            self.link_indices.append(plan.add(links[mu], link_displacements[mu]))

        self.site_fields_indices = []
        for i in range(self.n_site_fields):
            self.site_fields_indices.append(plan.add(site_fields[i], site_displacements))

        self.cshifts = plan()

    def __call__(self, links, site_fields=[]):
        assert len(site_fields) == self.n_site_fields
        assert len(links) == self.dim

        buffers = self.cshifts(links + site_fields)

        for p in self.paths:
            d = [0 for mu in range(self.dim)]
            r = None
            # pr = prof.Profile(timer=lambda: g.time()*100000.0)
            # pr.enable()
            for mu, distance in p.path:
                for step in range(abs(distance)):
                    factor = None
                    if distance > 0:
                        factor = buffers[self.link_indices[mu][tuple(d)]]
                    d[mu] += distance // abs(distance)
                    if distance < 0:
                        factor = g.adj(buffers[self.link_indices[mu][tuple(d)]])
                    assert factor is not None
                    if r is None:
                        r = factor
                    else:
                        r = r * factor
            # pr.disable()
            # pr.print_stats(sort="cumulative")
            assert r is not None
            if self.n_site_fields == 0:
                yield g.eval(r)
            else:
                yield g.eval(r), [
                    buffers[self.site_fields_indices[i][tuple(d)]]
                    for i in range(self.n_site_fields)
                ]
