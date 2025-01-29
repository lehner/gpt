#
#    GPT - Grid Python Toolkit
#    Copyright (C) 2020-25  Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)
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
from gpt.algorithms.integrator.symplectic.log import log
from gpt.algorithms.integrator.symplectic.base import symplectic_base
from gpt.algorithms.integrator.symplectic.update import (
    update_p,
    update_q,
    update_p_force_gradient,
    implicit_update,
)
from gpt.algorithms.integrator.symplectic.schemes import (
    leap_frog,
    OMF2,
    OMF2_force_gradient,
    OMF4,
    generic,
)
import gpt


def set_verbose(val=True):
    for i in ["leap_frog", "omf2", "omf2_force_gradient", "omf4"]:
        gpt.default.set_verbose(i, val)
