#
#    GPT - Grid Python Toolkit
#    Copyright (C) 2020  Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)
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
from gpt.qcd.fermion.preconditioner.g5m import g5m_ne

from gpt.qcd.fermion.preconditioner.even_odd_sites import (
    eo1,
    eo1_ne,
    eo2,
    eo2_ne,
    eo2_kappa_ne,
)

from gpt.qcd.fermion.preconditioner.sap import sap_cycle
from gpt.qcd.fermion.preconditioner.mixed_dwf import mixed_dwf
from gpt.qcd.fermion.preconditioner.physical import physical
