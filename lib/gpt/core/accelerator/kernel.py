#
#    GPT - Grid Python Toolkit
#    Copyright (C) 2025-26  Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)
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
import cgpt


class kernel:
    from gpt.core.accelerator.kernel_blas import gemm, inv, det
    from gpt.core.accelerator.kernel_core import accumulate, indexed_sum, contract, transpose
    from gpt.core.accelerator.kernel_comm import copy, expand_to_global, restrict_to_local
    from gpt.core.accelerator.kernel_fft import rank_fft, fft

    def __init__(self):
        self.obj = cgpt.create_kernel()
        self.references = []
        self.verbose = g.default.is_verbose("kernel")

    def __del__(self):
        cgpt.delete_kernel(self.obj)

    def __call__(self):
        if self.verbose:
            cgpt.timer_begin()
        cgpt.kernel_execute(self.obj)
        if self.verbose:
            t_cgpt = g.timer("cgpt_kernel_execute", True)
            t_cgpt += cgpt.timer_end()
            g.message(t_cgpt)
        return self

    def __str__(self):
        return cgpt.kernel_str(self.obj).strip()
