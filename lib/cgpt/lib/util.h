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
template<typename vtype>
void cgpt_ferm_to_prop(Lattice<iSpinColourVector<vtype>>& ferm, cgpt_Lattice_base* _prop, int s, int c, bool f2p) {
  Lattice<iSpinColourMatrix<vtype>> & prop = compatible<iSpinColourMatrix<vtype>>(_prop)->l;

  if (f2p) {
    for(int j = 0; j < Ns; j++) {
      auto pjs = peekSpin(prop, j, s);
      auto fj  = peekSpin(ferm, j);
      for(int i = 0; i < Nc; i++) {
	pokeColour(pjs, peekColour(fj,i), i, c);
      }
      pokeSpin(prop, pjs, j, s);
    }
  } else {
    for(int j = 0; j < Ns; j++) {
      auto pjs = peekSpin(prop, j, s);
      auto fj  = peekSpin(ferm, j);
      for(int i = 0; i < Nc; i++) {
	pokeColour(fj, peekColour(pjs, i,c),i);
      }
      pokeSpin(ferm, fj, j);
    }
  }

}

template<typename T>
void cgpt_ferm_to_prop(Lattice<T>& ferm, cgpt_Lattice_base* _prop, int s, int c, bool f2p) {
  ERR("not supported");
}

template<typename get_t, typename op_t, typename empty_t, typename red_t>
void cgpt_reduce_inplace(red_t & r, get_t get, op_t op, empty_t empty) {
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
auto cgpt_reduce(get_t get, op_t op, empty_t empty) -> typename std::remove_reference<decltype(get[0])>::type {

  typedef typename std::remove_reference<decltype(get[0])>::type red_t;

  red_t r;
  cgpt_reduce_inplace(r, get, [op](red_t & r, const red_t & a) {
				r = op(r, a);
			      }, empty);
  
  return r;
}

template<typename get_t, typename op_t, typename element_t>
void cgpt_partial_reduce(std::vector<element_t> & red, get_t get, op_t op) {
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
void cgpt_duplicate(std::vector<elem_t> & dup_data, std::vector<elem_t> & data, std::vector<ndup_t> & ndup,
		    modify_t mod) {
  std::vector<ndup_t> idx;
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
void cgpt_enumerate(std::vector<idx_t> & idx, get_t get, check_t check) {
  size_t n = get.size();
  if (!n) {
    idx.resize(0);
    return;
  }
  size_t threads;
  thread_region
    {
      threads = thread_max();
    }

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

  std::vector<size_t> ps_nt;
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

