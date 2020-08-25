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
class cgpt_block_map_base {
public:
  virtual ~cgpt_block_map_base() { };

  virtual void project(std::vector<cgpt_Lattice_base*>& _coarse, long _ncoarse,
		       std::vector<cgpt_Lattice_base*>& _fine, long _nfine) = 0;
  virtual void promote(std::vector<cgpt_Lattice_base*>& _coarse, long _ncoarse,
		       std::vector<cgpt_Lattice_base*>& _fine, long _nfine)  = 0;
  virtual void orthonormalize() = 0;
};

template<class T, class C>
class cgpt_block_map : public cgpt_block_map_base {

  typedef typename Lattice<T>::vector_object vobj;
  typedef typename vobj::scalar_object sobj;
  typedef typename Lattice<T>::vector_type vCoeff_t;
  typedef typename Lattice<T>::scalar_type Coeff_t;
  typedef iSinglet<vCoeff_t> T_singlet;

  std::vector<PVector<Lattice<T>>> vbasis;
  cgpt_block_lookup_table<T_singlet> lut;
  GridBase* coarse_grid;
  GridBase* fine_grid;

public:

  virtual ~cgpt_block_map() {
  }

  cgpt_block_map(GridBase* _coarse_grid, 
		 std::vector<std::vector<cgpt_Lattice_base*>>& _vbasis,
		 cgpt_Lattice_base* _mask) 
    :
    lut(_coarse_grid, compatible<T_singlet>(_mask)->l),
    coarse_grid(_coarse_grid),
    fine_grid(_vbasis[0][0]->get_grid())
  {
	
    vbasis.resize(_vbasis.size());
    for (long i=0;i<_vbasis.size();i++)
      cgpt_basis_fill(vbasis[i],_vbasis[i]);
  }

  virtual void project(std::vector<cgpt_Lattice_base*>& _coarse, long _ncoarse,
		       std::vector<cgpt_Lattice_base*>& _fine, long _nfine) {

    PVector<Lattice<T>> fine;
    PVector<Lattice<C>> coarse;
    cgpt_basis_fill(fine,_fine);
    cgpt_basis_fill(coarse,_coarse);

    ASSERT(_fine.size() == _nfine);
    ASSERT(_coarse.size() == _ncoarse);

    ASSERT(vbasis[0].size() % _ncoarse == 0);
    long coarse_block_size = vbasis[0].size() / _ncoarse;

    Lattice<C> tmp(coarse_grid);
    for (long i=0;i<_ncoarse;i++)
      coarse[i] = Zero();

    for (long j=0;j<_nfine;j++) {
      for (long i=0;i<_ncoarse;i++) {
	vectorizableBlockProjectUsingLut(tmp, fine[j], vbasis[j].slice(i*coarse_block_size,(i+1)*coarse_block_size), lut);
	coarse[i] += tmp;
      }
    }
  }

  virtual void promote(std::vector<cgpt_Lattice_base*>& _coarse, long _ncoarse,
		       std::vector<cgpt_Lattice_base*>& _fine, long _nfine) {


    PVector<Lattice<T>> fine;
    PVector<Lattice<C>> coarse;
    cgpt_basis_fill(fine,_fine);
    cgpt_basis_fill(coarse,_coarse);

    ASSERT(_fine.size() == _nfine);
    ASSERT(_coarse.size() == _ncoarse);

    ASSERT(vbasis[0].size() % _ncoarse == 0);
    long coarse_block_size = vbasis[0].size() / _ncoarse;

    Lattice<T> tmp(fine_grid);
    for (long i=0;i<_nfine;i++)
      fine[i] = Zero();

    for (long i=0;i<_ncoarse;i++) {
      for (long j=0;j<_nfine;j++) {
	blockPromote(coarse[i], tmp, vbasis[j].slice(i*coarse_block_size,(i+1)*coarse_block_size));
	fine[j] += tmp;
      }
    }
  }

  virtual void orthonormalize() {
    Lattice<T_singlet> c(coarse_grid);
    vectorBlockOrthonormalize(c,vbasis);
  }
};
