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
from gpt.qcd.honeycomb.geometry import geometry


class action:
    def __init__(self, geo):
        self.geo = geo

    from gpt.qcd.honeycomb.gauge.action import wilson


class gauge:
    def __init__(self, geo):
        self.geo = geo
        self.action = action(self.geo)

    from gpt.qcd.honeycomb.gauge import plaquette, transformed


class honeycomb:
    def __init__(self):
        self.geo = geometry()
        self.number_of_link_fields = self.geo.number_of_link_fields
        self.number_of_subsets = self.geo.number_of_subsets
        self.gauge = gauge(self.geo)
