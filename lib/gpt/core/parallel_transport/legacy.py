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


def parallel_transport(links, paths):
    # the transports along paths, as a function of the links (one
    # parallel_transport_matrix stencil)
    code = [(mu, -1, 1.0, paths[mu]) for mu in range(len(paths))]
    ptm = g.parallel_transport_matrix(links, code, len(paths))

    def _wrap(links):
        return g.util.to_list(ptm(links))

    return _wrap
