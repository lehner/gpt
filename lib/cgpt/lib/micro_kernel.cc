/*
    GPT - Grid Python Toolkit
    Copyright (C) 2020  Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)

    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License along
    with this program; if not, write to the Free Software Foundation, Inc.,
    51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
*/
#include "lib.h"

void eval_micro_kernels(const std::vector<micro_kernel_t> & kernels, const micro_kernel_blocking_t & blocking) {

  size_t n = kernels.size();

  size_t o_sites = kernels[0].arg.o_sites;
  size_t block_size = blocking.block_size;
  size_t subblock_size = blocking.subblock_size;

  micro_kernel_region({
      
      for (size_t j=0;j<(o_sites + block_size - 1)/block_size;j++) {

        for (size_t i=0;i<n;i++) {
          auto& k = kernels[i];
          
          size_t j0 = std::min(j*block_size, o_sites);
          size_t j1 = std::min(j0 + block_size, o_sites);
          k.action(k.arg, j0, j1, subblock_size);
        }

      }
    });
}
