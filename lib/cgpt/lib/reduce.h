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
template<typename get_t, typename op_t, typename empty_t, typename red_t>
void cgpt_reduce_inplace(red_t & r, const get_t & get, op_t op, empty_t empty) {
  size_t n = get.size();
  // TODO: thread reduction can be done in cost log_2(threads)
  r = empty;
  thread_region
    {
      red_t tr = empty;
      thread_for_in_region(i, n, {
	  op(tr, get[i]);
	});
      thread_critical
	{
	  op(r,tr);
	}
    }
}

template<typename get_t, typename op_t, typename empty_t>
auto cgpt_reduce(const get_t & get, op_t op, empty_t empty) -> typename std::remove_reference<decltype(get[0])>::type {

  typedef typename std::remove_const<typename std::remove_reference<decltype(get[0])>::type>::type red_t;

  red_t r;
  cgpt_reduce_inplace(r, get, [op](red_t & r, const red_t & a) {
				r = op(r, a);
			      }, empty);
  
  return r;
}

template<typename get_t, typename op_t, typename element_t>
void cgpt_partial_reduce(AlignedVector<element_t> & red, const get_t & get, op_t op) {
  size_t n = get.size();

  red.resize(n);
  if (!n)
    return;

  // TODO: make a parallel version
  red[0] = get[0];

  for (size_t i=1;i<n;i++)
    red[i] = op( red[i-1], get[i] );
}

template<typename elem_t, typename ndup_t, typename modify_t>
void cgpt_duplicate(AlignedVector<elem_t> & dup_data, AlignedVector<elem_t> & data, AlignedVector<ndup_t> & ndup,
		    modify_t mod) {

  AlignedVector<ndup_t> idx;

  cgpt_partial_reduce(idx, ndup, [](ndup_t a,ndup_t b){return a+b;});

  if (!idx.size())
    return;
  
  dup_data.resize(idx[idx.size()-1]);
  thread_for(i, data.size(), {
      size_t start = ( i > 0 ) ? idx[i-1] : 0;
      size_t end = idx[i];
      for (size_t j=start;j<end;j++) {
	dup_data[j] = data[i];
	mod(i, j-start, dup_data[j]);
      }
    });
}

template<typename idx_t, typename get_t, typename check_t>
void cgpt_enumerate(AlignedVector<idx_t> & idx, const get_t & get, check_t check) {
  size_t n = get.size();
  if (!n) {
    idx.resize(0);
    return;
  }
  size_t threads = thread_max();

  //  cgpt_timer t("enum");
  //t("parallel");
  
  size_t n_per_thread = (n + threads - 1) / threads;
  std::vector< std::vector< idx_t > > tidx(threads);
  std::vector<size_t> nt(threads);
  thread_region
    {
      std::vector<idx_t> tidx_me; // avoid thrashing
      size_t thread = thread_num();
      size_t start = n_per_thread * thread;
      size_t end = n_per_thread * (thread + 1);
      end = std::min(end, n);
      if (start < end) {
	for (size_t i=start;i<end;i++)
	  if (check(get[i]))
	    tidx_me.push_back(i);
      }
      tidx[thread] = std::move(tidx_me);
      nt[thread] = tidx[thread].size();
    }

  //t("pred");

  AlignedVector<size_t> ps_nt;
  cgpt_partial_reduce(ps_nt, nt, [](size_t a, size_t b){ return a+b; });

  //std::cout << GridLogMessage << "nt " << nt << std::endl;
  //std::cout << GridLogMessage << "ps_nt " << ps_nt << std::endl;

  //t("res");
  idx.resize(ps_nt[ps_nt.size()-1]);

  //t("final");
  thread_region
    {
      size_t thread = thread_num();
      size_t start = (thread == 0) ? 0 : ps_nt[thread-1];
      memcpy(&idx[start], &tidx[thread][0], nt[thread]*sizeof(size_t));
    }

  //t.report();
}

static long cgpt_gcd(long a, long b) {
  while (b) {
    long t = b;
    b = a % b;
    a = t;
  }
  return a;
}

template<typename check_t, typename vec_t, typename size_vec_t>
size_t cgpt_rle(size_vec_t & start, size_vec_t & repeats,
		const vec_t & a, check_t c) {
  
  AlignedVector<char> flags(a.size());
  thread_for(i, a.size()-1, { flags[i] = c(a[i],a[i+1]); });
  flags[a.size()-1] = 0;

  AlignedVector<size_t> idx;
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
