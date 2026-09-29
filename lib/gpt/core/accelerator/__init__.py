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
import cgpt
from gpt.core.accelerator.buffer_manager import buffer_manager
from gpt.core.accelerator.buffer import buffer
from gpt.core.accelerator.kernel import kernel


def backend():
    # accelerator backend of this build: "cuda", "hip", "sycl", or "none"
    return cgpt.accelerator_backend()


class threads:
    # temporarily set the threads per block of Grid's accelerator_for
    # (besides the SIMD lanes; Grid's default is --accelerator-threads);
    # n <= 0 keeps the current value
    def __init__(self, n):
        self.n = n

    def __enter__(self):
        self.previous = cgpt.accelerator_threads(self.n)
        return self

    def __exit__(self, *args):
        cgpt.accelerator_threads(self.previous)
