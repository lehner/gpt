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

// adj
template<typename T>
accelerator_inline typename std::enable_if<isGridTensor<T>::notvalue, void>::type cgpt_adj(T & r, const T & l) {
  coalescedWriteFundamental(r, conjugate(coalescedReadFundamental(l)));
}

template<typename T> accelerator_inline void cgpt_adj(iScalar<T> & r, const iScalar<T> & l);
template<typename T, int n> accelerator_inline void cgpt_adj(iMatrix<T,n> & r, const iMatrix<T,n> & l);
template<typename T, int n> accelerator_inline void cgpt_adj(iVector<T,n> & r, const iVector<T,n> & l);

template<typename T, int n> accelerator_inline void cgpt_adj(iMatrix<T,n> & r, const iMatrix<T,n> & l) {
  for (int i=0;i<n;i++)
    for (int j=0;j<n;j++)
      cgpt_adj(r(j,i), l(i,j));
}

template<typename T, int n> accelerator_inline void cgpt_adj(iVector<T,n> & r, const iVector<T,n> & l) {
  for (int i=0;i<n;i++)
    cgpt_adj(r(i), l(i));
}

template<typename T> accelerator_inline void cgpt_adj(iScalar<T> & r, const iScalar<T> & l) {
  cgpt_adj(r(), l());
}

// trans
template<typename T>
accelerator_inline typename std::enable_if<isGridTensor<T>::notvalue, void>::type cgpt_trans(T & r, const T & l) {
  coalescedWriteFundamental(r, coalescedReadFundamental(l));
}

template<typename T, int n> accelerator_inline void cgpt_trans(iMatrix<T,n> & r, const iMatrix<T,n> & l);
template<typename T, int n> accelerator_inline void cgpt_trans(iVector<T,n> & r, const iVector<T,n> & l);
template<typename T> accelerator_inline void cgpt_trans(iScalar<T> & r, const iScalar<T> & l);

template<typename T, int n> accelerator_inline void cgpt_trans(iMatrix<T,n> & r, const iMatrix<T,n> & l) {
  for (int i=0;i<n;i++)
    for (int j=0;j<n;j++)
      cgpt_trans(r(j,i), l(i,j));
}

template<typename T, int n> accelerator_inline void cgpt_trans(iVector<T,n> & r, const iVector<T,n> & l) {
  for (int i=0;i<n;i++)
    cgpt_trans(r(i), l(i));
}

template<typename T> accelerator_inline void cgpt_trans(iScalar<T> & r, const iScalar<T> & l) {
  cgpt_trans(r(), l());
}

// conj
template<typename T> accelerator_inline typename std::enable_if<isGridTensor<T>::notvalue, void>::type cgpt_conj(T & r, const T & l) {
  coalescedWriteFundamental(r, conjugate(coalescedReadFundamental(l)));
}

template<typename T, int n> accelerator_inline void cgpt_conj(iMatrix<T,n> & r, const iMatrix<T,n> & l);
template<typename T, int n> accelerator_inline void cgpt_conj(iVector<T,n> & r, const iVector<T,n> & l);
template<typename T> accelerator_inline void cgpt_conj(iScalar<T> & r, const iScalar<T> & l);

template<typename T, int n> accelerator_inline void cgpt_conj(iMatrix<T,n> & r, const iMatrix<T,n> & l) {
  for (int i=0;i<n;i++)
    for (int j=0;j<n;j++)
      cgpt_conj(r(i,j), l(i,j));
}

template<typename T, int n> accelerator_inline void cgpt_conj(iVector<T,n> & r, const iVector<T,n> & l) {
  for (int i=0;i<n;i++)
    cgpt_conj(r(i), l(i));
}

template<typename T> accelerator_inline void cgpt_conj(iScalar<T> & r, const iScalar<T> & l) {
  cgpt_conj(r(), l());
}


template<typename T>
void cgpt_unary(cgpt_Lattice<T> * pc, int unary) {

  if (!unary)
    return;

  GridBase* grid = pc->l.Grid();
  Lattice<T> r(grid);
  r.Checkerboard() = pc->l.Checkerboard();

  {
    autoView(r_v, r, AcceleratorWriteDiscard);
    autoView(l_v, pc->l, AcceleratorRead);
    
    auto p_r = &r_v[0];
    auto p_l = &l_v[0];
    
    switch (unary) {
    case BIT_TRANS|BIT_CONJ:
      accelerator_for(osite, grid->oSites(), grid->Nsimd(), {
	  cgpt_adj(p_r[osite], p_l[osite]);
	});
      break;
    case BIT_TRANS:
      accelerator_for(osite, grid->oSites(), grid->Nsimd(), {
	  cgpt_trans(p_r[osite], p_l[osite]);
	});
      break;
    case BIT_CONJ:
      accelerator_for(osite, grid->oSites(), grid->Nsimd(), {
	  cgpt_conj(p_r[osite], p_l[osite]);
	});
      break;
    }
  }

  pc->l = r;
}
