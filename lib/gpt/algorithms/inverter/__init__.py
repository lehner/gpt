#
#    GPT - Grid Python Toolkit
#    Copyright (C) 2020  Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)
#                  2020  Daniel Richtmann (daniel.richtmann@ur.de)
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
from gpt.algorithms.inverter.sequence import sequence
from gpt.algorithms.inverter.deflate import deflate
from gpt.algorithms.inverter.coarse_deflate import coarse_deflate
from gpt.algorithms.inverter.cg import cg
from gpt.algorithms.inverter.bicgstab import bicgstab
from gpt.algorithms.inverter.cagcr import cagcr
from gpt.algorithms.inverter.fom import fom
from gpt.algorithms.inverter.fgcr import fgcr
from gpt.algorithms.inverter.fgmres import fgmres
from gpt.algorithms.inverter.mr import mr
from gpt.algorithms.inverter.defect_correcting import defect_correcting
from gpt.algorithms.inverter.mixed_precision import mixed_precision
from gpt.algorithms.inverter.split import split
from gpt.algorithms.inverter.preconditioned import preconditioned
from gpt.algorithms.inverter.multi_grid import coarse_grid, multi_grid_setup
from gpt.algorithms.inverter.calculate_residual import calculate_residual
from gpt.algorithms.inverter.multi_shift import multi_shift
from gpt.algorithms.inverter.multi_shift_cg import multi_shift_cg
from gpt.algorithms.inverter.multi_shift_fom import multi_shift_fom
from gpt.algorithms.inverter.multi_shift_fgmres import multi_shift_fgmres
from gpt.algorithms.inverter.subspace_minimal_residual import subspace_minimal_residual
from gpt.algorithms.inverter.solution_history import solution_history
from gpt.algorithms.inverter.checkpointed import checkpointed
