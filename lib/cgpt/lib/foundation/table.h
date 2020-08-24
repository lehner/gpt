/*
    GPT - Grid Python Toolkit
    Copyright (C) 2020  Daniel Richtmann (daniel.richtmann@ur.de, https://github.com/lehner/gpt)
                  2020  Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)

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

class cgpt_lookup_table_base {
public:
  typedef uint64_t index_type;
  typedef uint64_t size_type;

  virtual ~cgpt_lookup_table_base() {};
  virtual accelerator_inline index_type const* const* View() const = 0;
  virtual accelerator_inline size_type const* Sizes() const = 0;
  virtual accelerator_inline index_type const* ReverseView() const = 0;
  virtual bool gridsMatch(GridBase* coarse, GridBase* fine) const = 0;
};

template<class ScalarField>
class cgpt_lookup_table : public cgpt_lookup_table_base {
  /////////////////////////////////////////////
  // Member Data
  /////////////////////////////////////////////

private:
  GridBase*                       coarse_;
  GridBase*                       fine_;
  std::vector<Vector<index_type>> lut_vec_;
  Vector<index_type*>             lut_ptr_;
  Vector<size_type>               sizes_;
  Vector<index_type>              reverse_lut_vec_;

  /////////////////////////////////////////////
  // Member Functions
  /////////////////////////////////////////////

public:

  cgpt_lookup_table(GridBase* coarse, ScalarField const& mask)
    : coarse_(coarse)
    , fine_(mask.Grid())
    , lut_vec_(coarse_->oSites())
    , lut_ptr_(coarse_->oSites())
    , sizes_(coarse_->oSites())
    , reverse_lut_vec_(fine_->oSites()){
    populate(coarse_, mask);
  }

  virtual ~cgpt_lookup_table() {
    // std::cout << "Deallocate" << std::endl;
  }

  virtual accelerator_inline
  std::vector<Vector<index_type>> const& operator()() const {
    return lut_vec_;
  } // CPU access (TODO: remove?)

  virtual accelerator_inline
  index_type const* const* View() const {
    return &lut_ptr_[0];
  } // GPU access

  virtual accelerator_inline
  size_type const* Sizes() const {
    return &sizes_[0];
  }  // also needed for GPU access

  virtual accelerator_inline
  index_type const* ReverseView() const {
    return &reverse_lut_vec_[0];
  }

  virtual bool gridsMatch(GridBase* coarse, GridBase* fine) const {
    return (coarse == coarse_) && (fine == fine_);
  }

private:
  void populate(GridBase* coarse, ScalarField const& mask) {
    int        _ndimension = coarse_->_ndimension;
    Coordinate block_r(_ndimension);

    size_type block_v = 1;
    for(int d = 0; d < _ndimension; ++d) {
      block_r[d] = fine_->_rdimensions[d] / coarse_->_rdimensions[d];
      assert(block_r[d] * coarse_->_rdimensions[d] == fine_->_rdimensions[d]);
      block_v *= block_r[d];
    }
    assert(block_v == fine_->oSites()/coarse_->oSites());

    lut_vec_.resize(coarse_->oSites());
    lut_ptr_.resize(coarse_->oSites());
    sizes_.resize(coarse_->oSites());
    reverse_lut_vec_.resize(fine_->oSites());
    for(index_type sc = 0; sc < coarse_->oSites(); ++sc) {
      lut_vec_[sc].resize(0);
      lut_vec_[sc].reserve(block_v);
      lut_ptr_[sc] = &lut_vec_[sc][0];
      sizes_[sc]  = 0;
    }

    typename ScalarField::scalar_type zz = {0., 0.,};

    autoView(mask_v, mask, CpuRead);
    thread_for(sc, coarse_->oSites(), {
      Coordinate coor_c(_ndimension);
      Lexicographic::CoorFromIndex(coor_c, sc, coarse_->_rdimensions);

      int sf_tmp, count = 0;
      for(int sb = 0; sb < block_v; ++sb) {
        Coordinate coor_b(_ndimension);
        Coordinate coor_f(_ndimension);

        Lexicographic::CoorFromIndex(coor_b, sb, block_r);
        for(int d = 0; d < _ndimension; ++d) coor_f[d] = coor_c[d] * block_r[d] + coor_b[d];
        Lexicographic::IndexFromCoor(coor_f, sf_tmp, fine_->_rdimensions);

        index_type sf = (index_type)sf_tmp;

        if(Reduce(TensorRemove(coalescedRead(mask_v[sf]))) != zz) {
          lut_ptr_[sc][count] = sf;
          sizes_[sc]++;
          count++;
        }
        reverse_lut_vec_[sf] = sc; // reverse table will never have holes
      }
      lut_vec_[sc].resize(sizes_[sc]);
    });

    std::cout << GridLogMessage << "Recalculation of coarsening lookup table finished" << std::endl;
  }
};

template<class vobj,class CComplex,int nbasis,class VLattice,class ScalarField>
inline void vectorizableBlockProjectUsingLut(Lattice<iVector<CComplex, nbasis>>&   coarseData,
                                             const Lattice<vobj>&                  fineData,
                                             const VLattice&                       Basis,
                                             const cgpt_lookup_table<ScalarField>& lut)
{
  GridBase *fine   = fineData.Grid();
  GridBase *coarse = coarseData.Grid();

  // checks
  assert(fine->_ndimension == coarse->_ndimension);
  assert(nbasis == Basis.size());
  assert(lut.gridsMatch(coarse, fine));

  auto lut_v = lut.View();
  auto sizes_v = lut.Sizes();
  autoView(fineData_v, fineData, AcceleratorRead);
  autoView(coarseData_v, coarseData, AcceleratorWrite);

  typedef decltype(Basis[0].View(AcceleratorRead)) View;
  Vector<View> Basis_v; Basis_v.reserve(Basis.size());
  for(int i=0;i<Basis.size();i++){
    Basis_v.push_back(Basis[i].View(AcceleratorRead));
  }

  accelerator_for(sci, nbasis*coarse->oSites(), vobj::Nsimd(), {
    auto i  = sci%nbasis;
    auto sc = sci/nbasis;

    decltype(innerProductD2(Basis_v[0](0), fineData_v(0))) reduce = Zero();

    for(int j=0; j<sizes_v[sc]; ++j) {
      int sf = lut_v[sc][j];
      reduce = reduce + innerProductD2(Basis_v[i](sf), fineData_v(sf));
    }
    convertType(coarseData_v[sc](i), TensorRemove(reduce));
  });

  for(int i=0;i<Basis.size();i++) Basis_v[i].ViewClose();
}

