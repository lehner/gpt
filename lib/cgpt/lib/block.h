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

template<class vobj>
inline void cgpt_blockSum(Lattice<vobj> &coarseData,const Lattice<vobj> &fineData) {

  GridBase * fine  = fineData.Grid();
  GridBase * coarse= coarseData.Grid();

  subdivides(coarse,fine); // require they map

  int _ndimension = coarse->_ndimension;

  Coordinate  block_r      (_ndimension);

  for(int d=0 ; d<_ndimension;d++){
    block_r[d] = fine->_rdimensions[d] / coarse->_rdimensions[d];
  }
  int blockVol = fine->oSites()/coarse->oSites();

  // Turn this around to loop threaded over sc and interior loop
  // over sf would thread better
  auto coarseData_ = coarseData.View();
  auto fineData_   = fineData.View();

  accelerator_for(sc,coarse->oSites(),1,{

      // One thread per sub block
      Coordinate coor_c(_ndimension);
      Lexicographic::CoorFromIndex(coor_c,sc,coarse->_rdimensions);  // Block coordinate
      coarseData_[sc]=Zero();

      for(int sb=0;sb<blockVol;sb++){

	int sf;
	Coordinate coor_b(_ndimension);
	Coordinate coor_f(_ndimension);
	Lexicographic::CoorFromIndex(coor_b,sb,block_r);               // Block sub coordinate
	for(int d=0;d<_ndimension;d++) coor_f[d]=coor_c[d]*block_r[d] + coor_b[d];
	Lexicographic::IndexFromCoor(coor_f,sf,fine->_rdimensions);

	coarseData_[sc]=coarseData_[sc]+fineData_[sf];
      }

    });
  return;
}

template<class vobj>
inline auto localInnerProductD(const Lattice<vobj> &lhs,const Lattice<vobj> &rhs)
-> Lattice<iScalar<decltype(TensorRemove(innerProductD2(lhs.View()(0),rhs.View()(0))))>>
{
  auto lhs_v = lhs.View();
  auto rhs_v = rhs.View();

  typedef decltype(TensorRemove(innerProductD2(lhs_v(0),rhs_v(0)))) t_inner;
  Lattice<iScalar<t_inner>> ret(lhs.Grid());
  auto ret_v = ret.View();

  accelerator_for(ss,rhs_v.size(),vobj::Nsimd(),{
      auto d2 = TensorPromote<1>::ToSinglet(TensorRemove(innerProductD2(lhs_v(ss),rhs_v(ss))));
      coalescedWrite(ret_v[ss],d2);
    });

  return ret;
}

template<class vobj,class CComplex>
  inline void cgpt_blockInnerProduct(Lattice<CComplex> &CoarseInner,
				      const Lattice<vobj> &fineX,
				      const Lattice<vobj> &fineY)
{
  typedef decltype(innerProduct(vobj(),vobj())) dotp;

  GridBase *coarse(CoarseInner.Grid());
  GridBase *fine  (fineX.Grid());

  Lattice<dotp> fine_inner(fine); fine_inner.Checkerboard() = fineX.Checkerboard();
  Lattice<dotp> coarse_inner(coarse);

  // Precision promotion?
  auto CoarseInner_  = CoarseInner.View();
  auto coarse_inner_ = coarse_inner.View();

  fine_inner = localInnerProduct(fineX,fineY);
  cgpt_blockSum(coarse_inner,fine_inner);
  accelerator_for(ss, coarse->oSites(), 1, {
      CoarseInner_[ss] = TensorRemove(coarse_inner_[ss]);
    });
}

static void precisionDemote(vComplexF & out, const vComplexD2 & in) {
  out.v = Optimization::PrecisionChange::DtoS(in._internal[0].v,in._internal[1].v);
}

static void precisionDemote(vComplexF & out, const vComplexF & in) {
  out = in;
}

static void precisionDemote(vComplexD & out, const vComplexD & in) {
  out = in;
}

template<typename vobj,typename T>
void precisionDemote(iScalar<vobj> & out, const T & in) {
  precisionDemote(out._internal,in);
}

template<class vobj,class CComplex>
  inline void cgpt_blockInnerProductD(Lattice<CComplex> &CoarseInner,
				      const Lattice<vobj> &fineX,
				      const Lattice<vobj> &fineY)
{
  typedef iScalar<decltype(TensorRemove(innerProductD2(vobj(),vobj())))> dotp;

  GridBase *coarse(CoarseInner.Grid());
  GridBase *fine  (fineX.Grid());

  Lattice<dotp> fine_inner(fine); fine_inner.Checkerboard() = fineX.Checkerboard();
  Lattice<dotp> coarse_inner(coarse);

  // Precision promotion?
  auto CoarseInner_  = CoarseInner.View();
  auto coarse_inner_ = coarse_inner.View();

  fine_inner = localInnerProductD(fineX,fineY);
  cgpt_blockSum(coarse_inner,fine_inner);
  accelerator_for(ss, coarse->oSites(), 1, {
      precisionDemote(CoarseInner_[ss], TensorRemove(coarse_inner_[ss]));
    });
 
}

template<class vobj,class CComplex>
  inline void cgpt_blockZAXPY(Lattice<vobj> &fineZ,
			      const Lattice<CComplex> &coarseA,
			      const Lattice<vobj> &fineX,
			      const Lattice<vobj> &fineY)
{
  GridBase * fine  = fineZ.Grid();
  GridBase * coarse= coarseA.Grid();

  fineZ.Checkerboard()=fineX.Checkerboard();
  assert(fineX.Checkerboard()==fineY.Checkerboard());
  subdivides(coarse,fine); // require they map
  conformable(fineX,fineY);
  conformable(fineX,fineZ);

  int _ndimension = coarse->_ndimension;

  Coordinate  block_r      (_ndimension);

  // FIXME merge with subdivide checking routine as this is redundant
  for(int d=0 ; d<_ndimension;d++){
    block_r[d] = fine->_rdimensions[d] / coarse->_rdimensions[d];
    assert(block_r[d]*coarse->_rdimensions[d]==fine->_rdimensions[d]);
  }

  auto fineZ_  = fineZ.View();
  auto fineX_  = fineX.View();
  auto fineY_  = fineY.View();
  auto coarseA_= coarseA.View();

  accelerator_for(sf, fine->oSites(), CComplex::Nsimd(), {

      int sc;
      Coordinate coor_c(_ndimension);
      Coordinate coor_f(_ndimension);

      Lexicographic::CoorFromIndex(coor_f,sf,fine->_rdimensions);
      for(int d=0;d<_ndimension;d++) coor_c[d]=coor_f[d]/block_r[d];
      Lexicographic::IndexFromCoor(coor_c,sc,coarse->_rdimensions);

      // z = A x + y
      coalescedWrite(fineZ_[sf],ConformSinglet(coarseA_(sc),vobj)*fineX_(sf)+fineY_(sf));

    });

  return;
}

template<class vobj,class CComplex,int nbasis>
  inline void cgpt_blockProject(Lattice<iVector<CComplex,nbasis > > &coarseData,
			   const             Lattice<vobj>   &fineData,
			   const std::vector<Lattice<vobj>* > &Basis)
{
  GridBase * fine  = fineData.Grid();
  GridBase * coarse= coarseData.Grid();

  Lattice<iScalar<CComplex>> ip(coarse);
  Lattice<vobj>     fineDataRed = fineData;

  //  auto fineData_   = fineData.View();
  auto coarseData_ = coarseData.View();
  auto ip_         = ip.View();
  for(int v=0;v<nbasis;v++) {
    cgpt_blockInnerProductD(ip,*Basis[v],fineDataRed); // ip = <basis|fine>
    accelerator_for( sc, coarse->oSites(), vobj::Nsimd(), {
	coalescedWrite(coarseData_[sc](v),TensorRemove(ip_(sc)));
      });

    // needed for numerical stability (crucial at single precision)
    // |fine> = |fine> - <basis|fine> |basis>
    ip=-ip;
    cgpt_blockZAXPY<vobj,iScalar<CComplex>> (fineDataRed,ip,*Basis[v],fineDataRed); 

  }
}


template<class vobj,class CComplex>
  inline void cgpt_blockNormalise(Lattice<CComplex> &ip,Lattice<vobj> &fineX)
{
  GridBase *coarse = ip.Grid();
  Lattice<vobj> zz(fineX.Grid()); zz=Zero(); zz.Checkerboard()=fineX.Checkerboard();
  cgpt_blockInnerProductD(ip,fineX,fineX);
  ip = pow(ip,-0.5);
  cgpt_blockZAXPY(fineX,ip,fineX,zz);
}

template<class vobj,class CComplex>
  inline void cgpt_blockOrthonormalize(Lattice<CComplex> &ip,std::vector<Lattice<vobj>* > &Basis)
{
  GridBase *coarse = ip.Grid();
  GridBase *fine   = Basis[0]->Grid();

  int       nbasis = Basis.size() ;

  // checks
  subdivides(coarse,fine);
  for(int i=0;i<nbasis;i++){
    conformable(Basis[i]->Grid(),fine);
  }

  for(int v=0;v<nbasis;v++) {
    for(int u=0;u<v;u++) {
      //Inner product & remove component
      cgpt_blockInnerProductD(ip,*Basis[u],*Basis[v]);
      ip = -ip;
      cgpt_blockZAXPY<vobj,CComplex> (*Basis[v],ip,*Basis[u],*Basis[v]);
    }
    cgpt_blockNormalise(ip,*Basis[v]);
  }
}

template<class vobj,class CComplex,int nbasis>
  inline void cgpt_blockPromote(const Lattice<iVector<CComplex,nbasis > > &coarseData,
			   Lattice<vobj>   &fineData,
			   const std::vector<Lattice<vobj>* > &Basis)
{
  GridBase * fine  = fineData.Grid();
  GridBase * coarse= coarseData.Grid();

  fineData=Zero();
  for(int i=0;i<nbasis;i++) {
    Lattice<iScalar<CComplex> > ip = PeekIndex<0>(coarseData,i);
    auto  ip_ =  ip.View();
    cgpt_blockZAXPY<vobj,iScalar<CComplex> >(fineData,ip,*Basis[i],fineData);
  }
}

template<typename T>
void cgpt_block_project(cgpt_Lattice_base* _coarse, Lattice<T>& fine, std::vector<cgpt_Lattice_base*>& _basis) {

  typedef typename Lattice<T>::vector_type vCoeff_t;

  std::vector< Lattice<T>* > basis(_basis.size());
  for (long i=0;i<_basis.size();i++)
    basis[i] = &compatible<T>(_basis[i])->l;

#define BASIS_SIZE(n) if (n == basis.size()) { cgpt_blockProject(compatible< iComplexV ## n<vCoeff_t> >(_coarse)->l,fine,basis); } else
#include "basis_size.h"
#undef BASIS_SIZE
  { ERR("Unknown basis size %d",(int)basis.size()); }

}



template<typename T>
void cgpt_block_promote(cgpt_Lattice_base* _coarse, Lattice<T>& fine, std::vector<cgpt_Lattice_base*>& _basis) {

  typedef typename Lattice<T>::vector_type vCoeff_t;

  std::vector< Lattice<T>* > basis(_basis.size());
  for (long i=0;i<_basis.size();i++)
    basis[i] = &compatible<T>(_basis[i])->l;

#define BASIS_SIZE(n) if (n == basis.size()) { cgpt_blockPromote(compatible< iComplexV ## n<vCoeff_t> >(_coarse)->l,fine,basis); } else
#include "basis_size.h"
#undef BASIS_SIZE
  { ERR("Unknown basis size %d",(int)basis.size()); }

}

template<typename T>
void cgpt_block_orthonormalize(cgpt_Lattice_base* _coarse, Lattice<T>& fine, std::vector<cgpt_Lattice_base*>& _basis) { // fine argument just to automatically detect type

  typedef typename Lattice<T>::vector_type vCoeff_t;

  std::vector< Lattice<T>* > basis(_basis.size());
  for (long i=0;i<_basis.size();i++)
    basis[i] = &compatible<T>(_basis[i])->l;

  cgpt_blockOrthonormalize(compatible< iSinglet<vCoeff_t> >(_coarse)->l,basis);
}
