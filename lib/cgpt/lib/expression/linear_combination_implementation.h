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

template<typename T1, int unary_expr, typename Enabled = void>
struct UnaryLinearCombination {
  typedef T1 result_type;
  static constexpr int n_elements = GridTypeMapper<result_type>::count;
  template<typename Accumulator, typename a_t, typename b_t>
  static accelerator_inline void eval(Accumulator & ac, uint64_t osite, a_t a, b_t b, int n, int e) {
    DEF_z();
    for (int i=0;i<n;i++)
      v += coalescedReadElement(a[i][osite], e) * b[i];
    ac.coalescedWriteElement(osite, v, e);
  }
};

template<typename T1, int ns>
struct UnaryLinearCombination<iScalar<iMatrix<iScalar<T1>,ns>>,BIT_SPINTRACE> {
  typedef iSinglet<T1> result_type;
  static constexpr int n_elements = 1;
  template<typename Accumulator, typename a_t, typename b_t>
  static accelerator_inline void eval(Accumulator & ac, uint64_t osite, a_t a, b_t b, int n, int e) {
    DEF_z();
    for (int i=0;i<n;i++)
      for (int j=0;j<ns;j++)
	v += coalescedReadElement(a[i][osite], j, j) * b[i];
    ac.coalescedWriteSinglet(osite, v);
  }
};

template<typename T1, int ns>
struct UnaryLinearCombination<iScalar<iMatrix<iScalar<T1>,ns>>,BIT_SPINTRACE|BIT_COLORTRACE> :
  UnaryLinearCombination<iScalar<iMatrix<iScalar<T1>,ns>>,BIT_SPINTRACE> { };


template<typename T1, int nc>
struct UnaryLinearCombination<iScalar<iScalar<iMatrix<T1,nc>>>,BIT_COLORTRACE> {
  typedef iSinglet<T1> result_type;
  static constexpr int n_elements = 1;
  template<typename Accumulator, typename a_t, typename b_t>
  static accelerator_inline void eval(Accumulator & ac, uint64_t osite, a_t a, b_t b, int n, int e) {
    DEF_z();
    for (int i=0;i<n;i++)
      for (int j=0;j<nc;j++)
	v += coalescedReadElement(a[i][osite], j, j) * b[i];
    ac.coalescedWriteSinglet(osite, v);
  }
};

template<typename T1, int nc>
struct UnaryLinearCombination<iScalar<iScalar<iMatrix<T1,nc>>>,BIT_SPINTRACE|BIT_COLORTRACE> :
  UnaryLinearCombination<iScalar<iScalar<iMatrix<T1,nc>>>,BIT_COLORTRACE> { };


template<typename T1, int ns, int nc>
struct UnaryLinearCombination<iScalar<iMatrix<iMatrix<T1,nc>,ns>>,BIT_SPINTRACE> {
  typedef iScalar<iScalar<iMatrix<T1,nc>>> result_type;
  static constexpr int n_elements = nc*nc;
  template<typename Accumulator, typename a_t, typename b_t>
  static accelerator_inline void eval(Accumulator & ac, uint64_t osite, a_t a, b_t b, int n, int e) {
    int ic = e / nc;
    int jc = e % nc;
    DEF_z();
    for (int i=0;i<n;i++)
      for (int j=0;j<ns;j++)
	v += coalescedReadElement(a[i][osite], j, j, ic, jc) * b[i];
    ac.coalescedWrite(osite, ic, jc, v);
  }
};

template<typename T1, int ns, int nc>
struct UnaryLinearCombination<iScalar<iMatrix<iMatrix<T1,nc>,ns>>,BIT_COLORTRACE> {
  typedef iScalar<iMatrix<iScalar<T1>,ns>> result_type;
  static constexpr int n_elements = ns*ns;
  template<typename Accumulator, typename a_t, typename b_t>
  static accelerator_inline void eval(Accumulator & ac, uint64_t osite, a_t a, b_t b, int n, int e) {
    int is = e / ns;
    int js = e % ns;
    DEF_z();
    for (int i=0;i<n;i++)
      for (int j=0;j<nc;j++)
	v += coalescedReadElement(a[i][osite], is, js, j, j) * b[i];
    ac.coalescedWrite(osite, is, js, v);
  }
};

template<typename T1, int ns, int nc>
struct UnaryLinearCombination<iScalar<iMatrix<iMatrix<T1,nc>,ns>>,BIT_COLORTRACE|BIT_SPINTRACE> {
  typedef iSinglet<T1> result_type;
  static constexpr int n_elements = 1;
  template<typename Accumulator, typename a_t, typename b_t>
  static accelerator_inline void eval(Accumulator & ac, uint64_t osite, a_t a, b_t b, int n, int e) {
    DEF_z();
    for (int i=0;i<n;i++)
      for (int js=0;js<ns;js++)
	for (int jc=0;jc<nc;jc++)
	  v += coalescedReadElement(a[i][osite], js, js, jc, jc) * b[i];
    ac.coalescedWriteSinglet(osite, v);
  }
};

template<typename T1, int N>
struct UnaryLinearCombination<iMatrix<iSinglet<T1>,N>,BIT_COLORTRACE> {
  typedef iSinglet<T1> result_type;
  static constexpr int n_elements = 1;
  template<typename Accumulator, typename a_t, typename b_t>
  static accelerator_inline void eval(Accumulator & ac, uint64_t osite, a_t a, b_t b, int n, int e) {
    DEF_z();
    for (int i=0;i<n;i++)
      for (int j=0;j<N;j++)
	v += coalescedReadElement(a[i][osite], j, j) * b[i];
    ac.coalescedWriteSinglet(osite, v);
  }
};

template<typename T1, int N>
struct UnaryLinearCombination<iMatrix<iSinglet<T1>,N>,BIT_SPINTRACE|BIT_COLORTRACE> :
  UnaryLinearCombination<iMatrix<iSinglet<T1>,N>,BIT_COLORTRACE> {};

template<typename T,int unary_expr,typename AccumulatorBase>
cgpt_Lattice_base* cgpt_lc(cgpt_Lattice_base* __c, std::vector<cgpt_lattice_term>& f, int unary_factor) {

  Timer("create lat");
  GridBase* grid = f[0].get_lat()->get_grid();

  typedef UnaryLinearCombination<T,unary_expr> U;
  typedef typename U::result_type R;
  typedef typename R::scalar_type Coeff_t;

  Lattice<R>* pc = 0;
  if (!__c) {
    __c = new cgpt_Lattice<R>(grid);
  }

  cgpt_Lattice<R> * _c = compatible<R>(__c);
  pc = &_c->l;

  int n = (int)f.size();
  Vector<Coeff_t> b(n);
  Vector<LatticeView<T>> v(n);
  Vector<T*> a(n);
  for (int i=0;i<n;i++) {
    b[i] = (Coeff_t)f[i].get_coef();
    v[i] = compatible<T>(f[i].get_lat())->l.View(AcceleratorRead);
    a[i] = &v[i][0];
  }
  
  Timer("view");

  pc->Checkerboard() = f[0].get_lat()->get_checkerboard();

  {
    autoView(c_v, (*pc), AcceleratorWrite);
  
    Accumulator<AccumulatorBase,R> ac(1.0,&c_v[0],&c_v[0]);

    auto p_b = &b[0];
    auto p_a = &a[0];

    Timer("loop");

    accelerator_for(ss, grid->oSites() * U::n_elements, grid->Nsimd(), {
	auto osite = ss / U::n_elements;
	auto j = ss - osite * U::n_elements;
	U::eval(ac, osite, p_a, p_b, n, j);
      });

    Timer();
  }

  for (int i=0;i<n;i++)
    v[i].ViewClose();

  return _c;

}

template<typename T,int unary_expr>
cgpt_Lattice_base* cgpt_lc(cgpt_Lattice_base* dst,bool ac, std::vector<cgpt_lattice_term>& f, int unary_factor) {

  typedef UnaryLinearCombination<T,unary_expr> U;
  typedef typename U::result_type R;

  if (unary_factor == 0) {
    if (ac) {
      return cgpt_lc<T,unary_expr,AccumulatorYesBase>(dst,f,0);
    } else {
      return cgpt_lc<T,unary_expr,AccumulatorNoBase>(dst,f,0);
    }
  } else {

    cgpt_Lattice_base* _lc = cgpt_lc<T,unary_expr,AccumulatorNoBase>(0,f,0);
    auto lc = compatible<R>(_lc);
    cgpt_unary(lc,unary_factor);

    if (ac) {

      compatible<R>(dst)->l += lc->l;
      delete lc;
      return dst;
      
    } else {
      if (dst) {
	compatible<R>(dst)->l = lc->l;
	delete lc;
	return dst;
      } else {
	return _lc;
      }
    }
    
  }
}

template<typename T>
cgpt_Lattice_base* cgpt_compatible_linear_combination(Lattice<T>& _compatible,cgpt_Lattice_base* dst,bool ac, std::vector<cgpt_lattice_term>& f, int unary_factor, int unary_expr) {
  if (unary_expr == 0) {
    return cgpt_lc<T,0>(dst,ac,f,unary_factor);
  } else if (unary_expr == BIT_COLORTRACE) {
    return cgpt_lc<T,BIT_COLORTRACE>(dst,ac,f,unary_factor);
  } else if (unary_expr == BIT_SPINTRACE) {
    return cgpt_lc<T,BIT_SPINTRACE>(dst,ac,f,unary_factor);
  } else if (unary_expr == (BIT_COLORTRACE|BIT_SPINTRACE)) {
    return cgpt_lc<T,BIT_COLORTRACE|BIT_SPINTRACE>(dst,ac,f,unary_factor);
  } else {
    ERR("Invalid unary_expr = %d",unary_expr);
  }
}

