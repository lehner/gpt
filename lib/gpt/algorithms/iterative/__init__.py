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
from gpt.algorithms.iterative.cg import cg
from gpt.algorithms.iterative.bicgstab import bicgstab
from gpt.algorithms.iterative.fgcr import fgcr
from gpt.algorithms.iterative.fgmres import fgmres
from gpt.algorithms.iterative.irl import irl
from gpt.algorithms.iterative.mr import mr
from gpt.algorithms.iterative.power_iteration import power_iteration
from gpt.algorithms.iterative.defect_correcting_inverter import (
    defect_correcting_inverter,
)
from gpt.algorithms.iterative.mg import mg
