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
template<class vobj,class CComplex,int nbasis,class VLattice>
  inline void vectorizableBlockProject(Lattice<iVector<CComplex,nbasis >> &coarseData,
				       const Lattice<vobj>   &fineData,
				       const VLattice &Basis)
{
  GridBase * fine  = fineData.Grid();
  GridBase * coarse= coarseData.Grid();

  Lattice<iScalar<CComplex>> ip(coarse);
  Lattice<vobj>     fineDataRed = fineData;

  //  auto fineData_   = fineData.View();
  auto coarseData_ = coarseData.AcceleratorView(ViewWrite);
  auto ip_         = ip.AcceleratorView(ViewReadWrite);
  for(int v=0;v<nbasis;v++) {
    blockInnerProductD(ip,Basis[v],fineDataRed); // ip = <basis|fine>
    accelerator_for( sc, coarse->oSites(), vobj::Nsimd(), {
	convertType(coarseData_[sc](v),ip_[sc]);
      });
  }
}

template<class CComplex,class VLattice>
inline void vectorBlockOrthonormalize(Lattice<CComplex> &ip,std::vector<VLattice> &Basis)
{
  GridBase *coarse = ip.Grid();
  GridBase *fine   = Basis[0][0].Grid();

  int       nvec = Basis.size();
  int       nbasis = Basis[0].size() ;

  // checks
  subdivides(coarse,fine);
  for (int j=0;j<nvec;j++){
    for(int i=0;i<nbasis;i++){
      conformable(Basis[j][i].Grid(),fine);
    }
  }

  Lattice<CComplex> sip(coarse);
  typename std::remove_reference<decltype(Basis[0][0])>::type zz(fine);
  zz = Zero();
  zz.Checkerboard()=Basis[0][0].Checkerboard();
  for(int v=0;v<nbasis;v++) {
    for(int u=0;u<v;u++) {
      //Inner product & remove component
      sip = Zero();
      for (int j=0;j<nvec;j++) {
	blockInnerProductD(ip,Basis[j][u],Basis[j][v]);
	sip -= ip;
      }
      for (int j=0;j<nvec;j++)
	blockZAXPY(Basis[j][v],sip,Basis[j][u],Basis[j][v]);
    }

    // block normalize
    sip = Zero();
    for (int j=0;j<nvec;j++) {
      blockInnerProductD(ip,Basis[j][v],Basis[j][v]);
      sip += ip;
    }
    sip = pow(sip,-0.5);
    ip = Zero();
    for (int j=0;j<nvec;j++)
      blockZAXPY(Basis[j][v],sip,Basis[j][v],zz);
  }
}

template<class vobj,class CComplex>
inline void blockMaskedInnerProductGPT(Lattice<CComplex> &coarseInner,
				    const Lattice<CComplex> &fineMask,
				    const Lattice<vobj> &fineX,
				    const Lattice<vobj> &fineY)
{
  GridBase * fine  = fineX.Grid();
  GridBase * coarse= coarseInner.Grid();
  int  _ndimension = coarse->_ndimension;

  // checks
  subdivides(coarse,fine);
  conformable(fineX, fineY);

  Coordinate block_r      (_ndimension);

  for(int d=0 ; d<_ndimension;d++){
    block_r[d] = fine->_rdimensions[d] / coarse->_rdimensions[d];
    assert(block_r[d]*coarse->_rdimensions[d] == fine->_rdimensions[d]);
  }
  int blockVol = fine->oSites()/coarse->oSites();

  coarseInner=Zero();
  typename CComplex::scalar_type zz = {0., 0.};

  auto fineX_v   = fineX.AcceleratorView(ViewRead);
  auto fineY_v   = fineY.AcceleratorView(ViewRead);
  auto fineMask_v   = fineMask.AcceleratorView(ViewRead);
  auto coarseInner_v = coarseInner.AcceleratorView(ViewWrite);

  accelerator_for( sc, coarse->oSites(), vobj::Nsimd(), {
    Coordinate coor_c(_ndimension);
    Lexicographic::CoorFromIndex(coor_c,sc,coarse->_rdimensions);  // Block coordinate

    int sf;
    decltype(innerProduct(fineX_v(sf),fineY_v(sf))) reduce=Zero();

    for(int sb=0;sb<blockVol;sb++){
      Coordinate coor_b(_ndimension);
      Coordinate coor_f(_ndimension);

      Lexicographic::CoorFromIndex(coor_b,sb,block_r);
      for(int d=0;d<_ndimension;d++) coor_f[d]=coor_c[d]*block_r[d]+coor_b[d];
      Lexicographic::IndexFromCoor(coor_f,sf,fine->_rdimensions);

      if(Reduce(TensorRemove(fineMask_v(sf))) != zz)
        reduce = reduce + innerProduct(fineX_v(sf),fineY_v(sf));
    }
    convertType(coarseInner_v[sc], TensorRemove(reduce));
  });
  return;
}


template<typename T>
void cgpt_block_project(cgpt_Lattice_base* _coarse, Lattice<T>& fine, std::vector<cgpt_Lattice_base*>& _basis) {

  typedef typename Lattice<T>::vector_type vCoeff_t;

  PVector<Lattice<T>> basis;
  cgpt_basis_fill(basis,_basis);

#define BASIS_SIZE(n) if (n == basis.size()) { vectorizableBlockProject(compatible< iVSinglet ## n<vCoeff_t> >(_coarse)->l,fine,basis); } else
#include "basis_size.h"
#undef BASIS_SIZE
  { ERR("Unknown basis size %d",(int)basis.size()); }

}



template<typename T>
void cgpt_block_promote(cgpt_Lattice_base* _coarse, Lattice<T>& fine, std::vector<cgpt_Lattice_base*>& _basis) {

  typedef typename Lattice<T>::vector_type vCoeff_t;

  PVector<Lattice<T>> basis;
  cgpt_basis_fill(basis,_basis);

#define BASIS_SIZE(n) if (n == basis.size()) { blockPromote(compatible< iVSinglet ## n<vCoeff_t> >(_coarse)->l,fine,basis); } else
#include "basis_size.h"
#undef BASIS_SIZE
  { ERR("Unknown basis size %d",(int)basis.size()); }

}

template<typename T>
void cgpt_block_orthonormalize(cgpt_Lattice_base* _coarse, Lattice<T>& fine, std::vector<std::vector<cgpt_Lattice_base*>>& _vbasis) { // fine argument just to automatically detect type

  typedef typename Lattice<T>::vector_type vCoeff_t;

  std::vector<PVector<Lattice<T>>> vbasis(_vbasis.size());
  for (long i=0;i<_vbasis.size();i++)
    cgpt_basis_fill(vbasis[i],_vbasis[i]);

  vectorBlockOrthonormalize(compatible< iSinglet<vCoeff_t> >(_coarse)->l,vbasis);
}

template<typename T>
void cgpt_block_masked_inner_product(cgpt_Lattice_base* _coarse, cgpt_Lattice_base* fineMask, Lattice<T>& fineX, Lattice<T>& fineY) {

  typedef typename Lattice<T>::vector_type vCoeff_t;

  blockMaskedInnerProductGPT(compatible< iSinglet<vCoeff_t> >(_coarse)->l,
                             compatible< iSinglet<vCoeff_t> >(fineMask)->l,
                             fineX, fineY);
}
