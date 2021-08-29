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
template<typename cmp_t, typename vec_t>
void cgpt_sort(vec_t& data, const cmp_t & cmp) {
  size_t threads = thread_max();
  size_t n = data.size();

  // split sorting into sub-blocks that will be pre-sorted per thread
  size_t n_per_thread = (n + threads - 1) / threads;
  thread_for(i, threads, {
      size_t start = i * n_per_thread;
      size_t end = (i+1)*n_per_thread;
      end = std::min(end, n);
      auto b = std::begin(data);
      if (end > start)
	std::sort(b + start, b + end, cmp);
    });

  // then merge two neighboring blocks at a time
  //std::cout << GridLogMessage << "threads " << threads << std::endl <<
  //  "n_per_thread " << n_per_thread << std::endl <<
  //  "n " << n << std::endl;

  // merge
  while (n_per_thread < n) {
    //double t0 = cgpt_time();
    thread_for(i, threads, {
	size_t start = std::min((size_t)(i * 2 * n_per_thread), n);
	size_t mid = std::min(start + n_per_thread, n);
	size_t end = std::min(mid + n_per_thread, n);
	auto b = std::begin(data);
	if (mid > start && end > mid)
	  std::inplace_merge( b+start, b+mid, b+end, cmp );
      });
    //double t1 = cgpt_time();
    n_per_thread *= 2;
    //std::cout << GridLogMessage << n_per_thread << " " << t1-t0 << std::endl;
  }


#if 0
  for (size_t i=1;i<n;i++) {
    if (!cmp(data[i-1],data[i])) {
      ERR("Sorting failed for %ld / %ld",i,n);
    }      
  }
#endif
}

template<typename check_t, typename vec_t>
void cgpt_sorted_unique(vec_t& u, const vec_t& a, check_t c) {
  AlignedVector<char> flags(a.size());
  thread_for(i, a.size()-1, { flags[i] = c(a[i],a[i+1]); });
  flags[a.size()-1] = 0;

  AlignedVector<size_t> idx;
  cgpt_enumerate(idx, flags, [](char x) { return !x; });

  u.resize(idx.size());
  thread_for(i, idx.size(), {
      u[i] = a[idx[i]];
    });
}

