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

template<typename Accumulator, int unary_expr, typename T1, typename T2, typename Enable = void>
cgpt_Lattice_base* cgpt_mul_acc_unary(cgpt_Lattice_base* _c,
				const Lattice<T1> & a,
				const Lattice<T2> & b) {
    
  GridBase* grid = a.Grid();

  typedef MultiplicationTable<T1,T2,Accumulator,unary_expr> MT;
  typedef typename MT::result_type T;

  if (!_c)
    _c = new cgpt_Lattice<T>(grid);

  auto & c = compatible<T>(_c)->l;
  c.Checkerboard() = a.Checkerboard();

  autoView(c_v, c, Accumulator::AcceleratorWriteMode);
  autoView(a_v, a, AcceleratorRead);
  autoView(b_v, b, AcceleratorRead);

  auto * p_c = &c_v[0];
  auto * p_a = &a_v[0];
  auto * p_b = &b_v[0];

  accelerator_for2d(osite, grid->oSites(), j, MT::n_elements, grid->Nsimd(), {
      MT::eval(p_c[osite], p_a[osite], p_b[osite], j);
    });

  return _c;
}

template<typename Accumulator, int unary_expr, typename T1, typename T2>
cgpt_Lattice_base* cgpt_mul_acc_unary(cgpt_Lattice_base* _c,
				      const Lattice<T1> & a,
				      const T2 & b) {
    
  GridBase* grid = a.Grid();

  typedef MultiplicationTable<T1,T2,Accumulator,unary_expr> MT;
  typedef typename MT::result_type T;

  if (!_c)
    _c = new cgpt_Lattice<T>(grid);

  auto & c = compatible<T>(_c)->l;
  c.Checkerboard() = a.Checkerboard();

  autoView(c_v, c, Accumulator::AcceleratorWriteMode);
  autoView(a_v, a, AcceleratorRead);

  auto * p_c = &c_v[0];
  auto * p_a = &a_v[0];
  auto * p_b = &b;

  accelerator_for2d(osite, grid->oSites(), j, MT::n_elements, grid->Nsimd(), {
      MT::eval(p_c[osite], p_a[osite], *p_b, j);
    });

  return _c;
}

template<typename Accumulator, int unary_expr, typename T1, typename T2>
cgpt_Lattice_base* cgpt_mul_acc_unary(cgpt_Lattice_base* _c,
				      const T1 & a,
				      const Lattice<T2> & b) {

  GridBase* grid = b.Grid();

  typedef MultiplicationTable<T1,T2,Accumulator,unary_expr> MT;
  typedef typename MT::result_type T;

  if (!_c)
    _c = new cgpt_Lattice<T>(grid);

  auto & c = compatible<T>(_c)->l;
  c.Checkerboard() = b.Checkerboard();

  autoView(c_v, c, Accumulator::AcceleratorWriteMode);
  autoView(b_v, b, AcceleratorRead);

  auto * p_c = &c_v[0];
  auto * p_a = &a;
  auto * p_b = &b_v[0];

  accelerator_for2d(osite, grid->oSites(), j, MT::n_elements, grid->Nsimd(), {
      MT::eval(p_c[osite], *p_a, p_b[osite], j);
    });

  return _c;
}

template<typename Accumulator, typename T1, typename T2>
cgpt_Lattice_base* cgpt_mul_acc(cgpt_Lattice_base* _c,
				const T1 & a,
				const T2 & b,
				int unary_expr) {

  switch (unary_expr) {
  case 0:
    return cgpt_mul_acc_unary<Accumulator,0>(_c,a,b);
  case BIT_SPINTRACE:
    return cgpt_mul_acc_unary<Accumulator,BIT_SPINTRACE>(_c,a,b);
  case BIT_COLORTRACE:
    return cgpt_mul_acc_unary<Accumulator,BIT_COLORTRACE>(_c,a,b);
  case BIT_COLORTRACE|BIT_SPINTRACE:
    return cgpt_mul_acc_unary<Accumulator,BIT_SPINTRACE|BIT_COLORTRACE>(_c,a,b);
  default:
    ERR("Unknown unary %d", unary_expr);
  }
}

template<typename T1, typename T2>
cgpt_Lattice_base* cgpt_mul(cgpt_Lattice_base* _c, bool ac,
			    const T1 & a,
			    const T2 & b,
			    int unary_expr) {

  if (ac) {
    ASSERT(_c);
    return cgpt_mul_acc<AccumulatorYes>(_c,a,b,unary_expr);
  } else {
    return cgpt_mul_acc<AccumulatorNo>(_c,a,b,unary_expr);
  }
}

template<typename T>
const Lattice<T> cgpt_unary(int unary, const Lattice<T> & l) {
  switch (unary) {
  case 0:
    return l;
  case BIT_TRANS|BIT_CONJ:
    return adj(l);
  case BIT_TRANS:
    return transpose(l);
  case BIT_CONJ:
    return conjugate(l);
  default:
    ERR("Unknown unary %d",unary);
  }
}

template<typename A, typename B>
  cgpt_Lattice_base* lattice_mul(cgpt_Lattice_base* dst, bool ac, int unary_a, const A& la, int unary_b, const B& lb,int unary_expr) {
  ASSERT(la.Grid() == lb.Grid());
  return cgpt_mul(dst, ac, cgpt_unary(unary_a, la), cgpt_unary(unary_b, lb), unary_expr);
}

template<typename A, typename B>
  cgpt_Lattice_base* lattice_unary_mul(cgpt_Lattice_base* dst, bool ac, int unary_a, const A& la, const B& ab,int unary_expr) {
  return cgpt_mul(dst, ac, cgpt_unary(unary_a, la), ab, unary_expr);
}

template<typename A, typename B>
  cgpt_Lattice_base* lattice_unary_rmul(cgpt_Lattice_base* dst, bool ac, int unary_a, const A& la, const B& ab,int unary_expr) {
  return cgpt_mul(dst, ac, ab, cgpt_unary(unary_a, la), unary_expr);
}
