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

// Product loops with the factor unaries applied on read (cgpt_unary_ref),
// only for unary_expr == 0 (traced products with unary factors use
// temporaries).  Selected by tag dispatch rather than if-constexpr, which CUDA
// extended lambdas reject.
template<typename MT, typename T1, typename T2, typename Ac>
void cgpt_mul_unary_loop(std::true_type, GridBase* grid, Ac & ac, const T1* p_a, const T2* p_b, int unary_a, int unary_b) {
#ifndef GRID_HAS_ACCELERATOR
  accelerator_for(osite, grid->oSites(), (size_t)grid->Nsimd(), {
      PREFETCH(p_a[osite]);
      PREFETCH(p_b[osite]);
      cgpt_unary_ref<T1> ra{p_a[osite], unary_a};
      cgpt_unary_ref<T2> rb{p_b[osite], unary_b};
      for (int j=0;j<MT::n_elements;j++) {
	MT::eval(ac, osite, ra, rb, j);
      }
    });
#else
  accelerator_for(ss, grid->oSites() * MT::n_elements, (size_t)grid->Nsimd(), {
      auto osite = ss / MT::n_elements;
      auto j = ss - osite * MT::n_elements;
      cgpt_unary_ref<T1> ra{p_a[osite], unary_a};
      cgpt_unary_ref<T2> rb{p_b[osite], unary_b};
      MT::eval(ac, osite, ra, rb, j);
    });
#endif
}

template<typename MT, typename T1, typename T2, typename Ac>
void cgpt_mul_unary_loop(std::false_type, GridBase* grid, Ac & ac, const T1* p_a, const T2* p_b, int unary_a, int unary_b) {
  ERR("Unary factors in a traced product are not applied on read");
}

template<typename MT, typename T1, typename T2, typename Ac>
void cgpt_mul_unary_const_loop(std::true_type, GridBase* grid, Ac & ac, const T1* p_a, const T2* p_b, int unary_a) {
#ifndef GRID_HAS_ACCELERATOR
  accelerator_for(osite, grid->oSites(), (size_t)grid->Nsimd(), {
      PREFETCH(p_a[osite]);
      cgpt_unary_ref<T1> ra{p_a[osite], unary_a};
      for (int j=0;j<MT::n_elements;j++) {
	MT::eval(ac, osite, ra, *p_b, j);
      }
    });
#else
  accelerator_for(ss, grid->oSites() * MT::n_elements, (size_t)grid->Nsimd(), {
      auto osite = ss / MT::n_elements;
      auto j = ss - osite * MT::n_elements;
      cgpt_unary_ref<T1> ra{p_a[osite], unary_a};
      MT::eval(ac, osite, ra, *p_b, j);
    });
#endif
}

template<typename MT, typename T1, typename T2, typename Ac>
void cgpt_mul_unary_const_loop(std::false_type, GridBase* grid, Ac & ac, const T1* p_a, const T2* p_b, int unary_a) {
  ERR("Unary factors in a traced product are not applied on read");
}

// unary_a/unary_b (BIT_TRANS|BIT_CONJ) are applied to the operands on read
// (cgpt_unary_ref), without temporaries; only for unary_expr == 0
template<typename AccumulatorBase, int unary_expr, bool rev, typename T1, typename T2, typename Enable = void>
cgpt_Lattice_base* cgpt_mul_acc_unary(cgpt_Lattice_base* _c,
				      const Lattice<T1> & a,
				      const Lattice<T2> & b,
				      ComplexD coef, int unary_a = 0, int unary_b = 0) {

  Timer("create lat");
  GridBase* grid = a.Grid();

  ASSERT(rev == false);

  typedef MultiplicationTable<T1,T2,unary_expr> MT;
  typedef typename MT::result_type T;

  cgpt_Lattice_base* _d = 0;

  // d = a*b + c
  Lattice<T>* pd = 0;
  Lattice<T>* pc = 0;
  if (!_c) {
    _c = new cgpt_Lattice<T>(grid);
    _d = _c;
    pc = &compatible<T>(_c)->l;
    pd = pc;
  } else {
    pc = &compatible<T>(_c)->l;
    if ((void*)pc == (void*)&a || (void*)pc == (void*)&b) {
      // do not overwrite existing memory in this case, a guarantee for MT to allow for faster implementations
      _d = new cgpt_Lattice<T>(grid);
    } else {
      _d = _c;
    }
    pd = &compatible<T>(_d)->l;
  }
  Timer("view");

  pd->Checkerboard() = a.Checkerboard();

  {
    autoView(a_v, a, AcceleratorRead);
    autoView(b_v, b, AcceleratorRead);
    autoView(c_v, (*pc), AcceleratorRead);
    autoView(d_v, (*pd), AcceleratorWrite);
  
    auto * p_a = &a_v[0];
    auto * p_b = &b_v[0];

    Accumulator<AccumulatorBase,T> ac(coef,&c_v[0],&d_v[0]);

    Timer("loop");
    
    if (unary_a || unary_b) {
      cgpt_mul_unary_loop<MT,T1,T2>(std::integral_constant<bool, unary_expr == 0>(), grid, ac, p_a, p_b, unary_a, unary_b);
    } else {
    // TODO: if c==a or c==b, need to copy to new memory region
#ifndef GRID_HAS_ACCELERATOR
    accelerator_for(osite, grid->oSites(), (size_t)grid->Nsimd(), {
	PREFETCH(p_a[osite]);
	PREFETCH(p_b[osite]);
	for (int j=0;j<MT::n_elements;j++) {
	  MT::eval(ac, osite, p_a[osite], p_b[osite], j);
	}
      });
#else
    accelerator_for(ss, grid->oSites() * MT::n_elements, (size_t)grid->Nsimd(), {
	auto osite = ss / MT::n_elements;
	auto j = ss - osite * MT::n_elements;
	MT::eval(ac, osite, p_a[osite], p_b[osite], j);
      });
#endif
    }

    Timer();
  }

  if (_c != _d)
    delete _c;
  return _d;
}

template<typename AccumulatorBase, int unary_expr, bool rev, typename T1, typename T2>
cgpt_Lattice_base* cgpt_mul_acc_unary(cgpt_Lattice_base* _c,
				      const Lattice<T1> & a,
				      const T2 & b,
				      ComplexD coef, int unary_a = 0, int unary_b = 0) {
    
  GridBase* grid = a.Grid();

  typedef MultiplicationTableRev<T1,T2,unary_expr,rev> MT;
  typedef typename MT::result_type T;

  cgpt_Lattice_base* _d = 0;

  // d = a*b + c
  Lattice<T>* pd = 0;
  Lattice<T>* pc = 0;
  if (!_c) {
    _c = new cgpt_Lattice<T>(grid);
    _d = _c;
    pc = &compatible<T>(_c)->l;
    pd = pc;
  } else {
    pc = &compatible<T>(_c)->l;
    if ((void*)pc == (void*)&a) {
      // do not overwrite existing memory in this case, a guarantee for MT to allow for faster implementations
      _d = new cgpt_Lattice<T>(grid);
    } else {
      _d = _c;
    }
    pd = &compatible<T>(_d)->l;
  }
  
  pd->Checkerboard() = a.Checkerboard();

  {
    autoView(a_v, a, AcceleratorRead);
    autoView(c_v, (*pc), AcceleratorRead);
    autoView(d_v, (*pd), AcceleratorWrite);
    
    Accumulator<AccumulatorBase,T> ac(coef, &c_v[0], &d_v[0]);
    auto * p_a = &a_v[0];
    
    static HostDeviceVector<T2> v_b(1);
    v_b[0] = b;
    
    auto * p_b = v_b.toDevice();
    
    ASSERT(!unary_b);
    if (unary_a) {
      cgpt_mul_unary_const_loop<MT,T1,T2>(std::integral_constant<bool, unary_expr == 0>(), grid, ac, p_a, p_b, unary_a);
    } else {
#ifndef GRID_HAS_ACCELERATOR
    accelerator_for(osite, grid->oSites(), (size_t)grid->Nsimd(), {
	PREFETCH(p_a[osite]);
	PREFETCH(p_b[osite]);
	for (int j=0;j<MT::n_elements;j++) {
	  MT::eval(ac, osite, p_a[osite], *p_b, j);
	}
      });
#else
    accelerator_for(ss, grid->oSites() * MT::n_elements, (size_t)grid->Nsimd(), {
	auto osite = ss / MT::n_elements;
	auto j = ss - osite * MT::n_elements;
	MT::eval(ac, osite, p_a[osite], *p_b, j);
      });
#endif
    }
  }

  if (_d != _c)
    delete _c;
  return _d;
}

template<typename AccumulatorBase, bool rev, typename T1, typename T2>
cgpt_Lattice_base* cgpt_mul_acc(cgpt_Lattice_base* _c,
				const T1 & a,
				const T2 & b,
				int unary_expr,
				ComplexD coef, int unary_a, int unary_b) {

  switch (unary_expr) {
  case 0:
    return cgpt_mul_acc_unary<AccumulatorBase,0,rev>(_c,a,b,coef,unary_a,unary_b);
  case BIT_SPINTRACE:
    return cgpt_mul_acc_unary<AccumulatorBase,BIT_SPINTRACE,rev>(_c,a,b,coef,unary_a,unary_b);
  case BIT_COLORTRACE:
    return cgpt_mul_acc_unary<AccumulatorBase,BIT_COLORTRACE,rev>(_c,a,b,coef,unary_a,unary_b);
  case BIT_COLORTRACE|BIT_SPINTRACE:
    return cgpt_mul_acc_unary<AccumulatorBase,BIT_SPINTRACE|BIT_COLORTRACE,rev>(_c,a,b,coef,unary_a,unary_b);
  default:
    ERR("Unknown unary %d", unary_expr);
  }
}

template<bool rev, typename T1, typename T2>
cgpt_Lattice_base* cgpt_mul(cgpt_Lattice_base* _c, bool ac,
			    const T1 & a,
			    const T2 & b,
			    int unary_expr,
			    ComplexD coef, int unary_a = 0, int unary_b = 0) {

  if (ac) {
    ASSERT(_c);
    return cgpt_mul_acc<AccumulatorYesBase,rev>(_c,a,b,unary_expr,coef,unary_a,unary_b);
  } else {
    return cgpt_mul_acc<AccumulatorNoBase,rev>(_c,a,b,unary_expr,coef,unary_a,unary_b);
  }
}


template<typename T>
void cgpt_unary(const Lattice<T> * & pl, const Lattice<T> & l, int unary) {
  // TODO: replace this by creating a left-hand-side/right-hand-side Reader template class
  // that can switch indices / CC in coalescedRead...
  switch (unary) {
  case 0:
    pl = &l;
    break;
  case BIT_TRANS|BIT_CONJ:
    pl = new Lattice<T>( transpose(conjugate(l)) ); // temporary fix
    break;
  case BIT_TRANS:
    pl = new Lattice<T>( transpose(l) );
    break;
  case BIT_CONJ:
    pl = new Lattice<T>( conjugate(l) );
    break;
  default:
    ERR("Unknown unary %d",unary);
  }
}

template<typename A, typename B>
cgpt_Lattice_base* lattice_mul(cgpt_Lattice_base* dst, bool ac, int unary_a, const A& la, int unary_b, const B& lb,int unary_expr,ComplexD coef) {
  ASSERT(la.Grid() == lb.Grid());
  Timer("lattice_mul");
  if (unary_expr == 0) {
    // unary factors applied on read (no temporaries)
    cgpt_Lattice_base* ret = cgpt_mul<false>(dst, ac, la, lb, 0, coef, unary_a, unary_b);
    Timer();
    return ret;
  }
  const A * pa;
  const B * pb;
  cgpt_unary(pa,la,unary_a);
  cgpt_unary(pb,lb,unary_b);
  cgpt_Lattice_base* ret = cgpt_mul<false>(dst, ac, *pa, *pb, unary_expr,coef);
  if (pa != &la)
    delete pa;
  if (pb != &lb)
    delete pb;
  Timer();
  return ret;
}

template<typename A, typename B>
cgpt_Lattice_base* lattice_unary_mul(cgpt_Lattice_base* dst, bool ac, int unary_a, const A& la, const B& ab,int unary_expr,ComplexD coef) {
  if (unary_expr == 0)
    return cgpt_mul<false>(dst, ac, la, ab, 0, coef, unary_a, 0);
  const A * pa;
  cgpt_unary(pa,la,unary_a);
  cgpt_Lattice_base* ret = cgpt_mul<false>(dst, ac, *pa, ab, unary_expr,coef);
  if (pa != &la)
    delete pa;
  return ret;
}

template<typename A, typename B>
cgpt_Lattice_base* lattice_unary_rmul(cgpt_Lattice_base* dst, bool ac, int unary_a, const A& la, const B& ab,int unary_expr, ComplexD coef) {
  if (unary_expr == 0)
    return cgpt_mul<true>(dst, ac, la, ab, 0, coef, unary_a, 0);
  const A * pa;
  cgpt_unary(pa,la,unary_a);
  cgpt_Lattice_base* ret = cgpt_mul<true>(dst, ac, *pa, ab, unary_expr,coef);
  if (pa != &la)
    delete pa;
  return ret;
}
