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
template<typename cmp_t, typename element_t>
void cgpt_sort(std::vector<element_t>& data, const cmp_t & cmp) {
  size_t threads;
  size_t n = data.size();
  thread_region
    {
      threads = thread_max();
    }

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
	size_t start = std::min(i * 2 * n_per_thread, n);
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

template<typename check_t, typename element_t>
void cgpt_sorted_unique(std::vector<element_t>& u, const std::vector<element_t>& a, check_t c) {
  std::vector<char> flags(a.size());
  thread_for(i, a.size()-1, { flags[i] = c(a[i],a[i+1]); });
  flags[a.size()-1] = 0;

  std::vector<size_t> idx;
  cgpt_enumerate(idx, flags, [](char x) { return !x; });

  u.resize(idx.size());
  thread_for(i, idx.size(), {
      u[i] = a[idx[i]];
    });
}

static long cgpt_gcd(long a, long b) {
  while (b) {
    long t = b;
    b = a % b;
    a = t;
  }
  return a;
}

template<typename check_t, typename element_t>
size_t cgpt_rle(std::vector<size_t> & start, std::vector<size_t> & repeats,
		const std::vector<element_t> & a, check_t c) {
  
  std::vector<char> flags(a.size());
  thread_for(i, a.size()-1, { flags[i] = c(a[i],a[i+1]); });
  flags[a.size()-1] = 0;

  std::vector<size_t> idx;
  cgpt_enumerate(idx, flags, [](char x) { return !x; });

  start.resize(idx.size());
  repeats.resize(idx.size());

  // Example 1:
  // ABCDE
  // 11110 flags
  // 4 idx
  // start = { 0 }, repeats = { 5 }

  // Example 2:
  // ACEGH
  // 00000 flags
  // {0,1,2,3,4} idx
  // start   = { 0, 1, 2, 3, 4 }
  // repeats = { 1, 1, 1, 1, 1 }

  // Example 3:
  // ABCEFG
  // 110110 flags
  // idx = { 2, 5 }
  // start   = { 0, 3 }
  // repeats = { 3, 3 }
  size_t gcd = 0;
  thread_region
    {
      size_t tgcd = 0;
      
      thread_for_in_region(i, idx.size(), {

	  // start[i] + repeats[i] == start[i+1]
	  // idx[-1] = -1
	  if (i) {
	    start[i] = idx[i-1] + 1;
	    repeats[i] = idx[i] - idx[i-1];
	    tgcd = (size_t)cgpt_gcd((long)tgcd, (long)repeats[i]);
	  } else {
	    start[0] = 0;
	    repeats[0] = idx[0] + 1;
	    tgcd = repeats[0];
	  }
	});

      thread_critical
	{
	  if (tgcd) {
	    if (gcd) {
	      gcd = (size_t)cgpt_gcd((long)gcd, (long)tgcd);
	    } else {
	      gcd = tgcd;
	    }
	  }
	}
    }

  return gcd;
}
