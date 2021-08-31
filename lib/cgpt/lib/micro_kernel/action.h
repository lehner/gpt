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

typedef void (* micro_kernel_action_t)(const micro_kernel_arg_t & arg, size_t i0, size_t i1, size_t subblock_size);

struct micro_kernel_t {
  micro_kernel_action_t action;
  micro_kernel_arg_t arg;
};

struct micro_kernel_blocking_t {
  size_t block_size;
  size_t subblock_size;
};

void eval_micro_kernels(const std::vector<micro_kernel_t> & kernels, const micro_kernel_blocking_t & blocking);
