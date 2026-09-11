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


def matrix(self, *fields):
    if self.write_fields is None:
        raise Exception(
            "Generalized matrix stencil needs more information.  Call stencil.data_access_hints."
        )
    if self.verbose_performance:
        t = g.timer("stencil.matrix")
        t("create fields")
    padded_fields = []
    padded_field = None
    for i in range(len(fields)):
        if i in self.read_fields:
            padded_field = self.padding(fields[i])
            padded_fields.append(padded_field)
        else:
            padded_fields.append(None)
    assert padded_field is not None
    for i in range(len(fields)):
        if padded_fields[i] is None:
            # start from the caller's current value (not fresh scratch)
            # so that accumulate entries build on it; fresh-write
            # entries (accumulate=-1) overwrite it anyway
            padded_fields[i] = self.padding(fields[i])
    if self.verbose_performance:
        t("local stencil")
    self.local_stencil(*padded_fields)
    if self.verbose_performance:
        t("extract")
    for i in self.write_fields:
        self.padding.extract(fields[i], padded_fields[i])
    if self.verbose_performance:
        t()
        g.message(t)
    # todo: make use of cache_fields
