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
#ifndef GRID_HAS_ACCELERATOR

#define micro_kernel_for(idx, n_idx, nsimd, nsubblock, ...) {           \
    int n_thread = thread_num();                                        \
    int n_threads = thread_max();                                       \
    for (size_t ib=n_subblock*n_thread;ib<n_idx;ib+=n_subblock*n_threads) { \
      for (size_t idx=ib;idx<ib+n_subblock && idx<n_idx;idx++) {        \
        __VA_ARGS__;                                                    \
      }}}
#define micro_kernel_region(...) { thread_region { __VA_ARGS__ } }

#else

// TODO: for2d and use nsubblock
#define micro_kernel_for(idx, n_idx, nsimd, nsubblock, ...) accelerator_forNB(idx, n_idx, nsimd, __VA_ARGS__)
//#define micro_kernel_for(idx, n_idx, nsimd, nsubblock, ...) accelerator_for2dNB(ib, n_idx / nsubblock, jb, nsubblock, nsimd, uint64_t idx=ib__VA_ARGS__)
#define micro_kernel_region(...) { __VA_ARGS__; accelerator_barrier(dummy); }

#endif

#define micro_kernel_view(vobj, ptr, idx)                               \
  auto ptr ## _v = ((ViewContainer<LatticeView<vobj>>*)arg.views[idx].view)->v; \
  auto ptr = &ptr ## _v[arg.views[idx].persistent ? i0 : 0];

