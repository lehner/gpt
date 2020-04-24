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


accelerator_inline void convertType(vComplexF & out, const vComplexD2 & in) {
  out.v = Optimization::PrecisionChange::DtoS(in._internal[0].v,in._internal[1].v);
}

accelerator_inline void convertType(vComplexD2 & out, const vComplexF & in) {
  Optimization::PrecisionChange::StoD(in.v,out._internal[0].v,out._internal[1].v);
}

template<typename T1,typename T2>
accelerator_inline void convertType(iScalar<T1> & out, const T2 & in) {
  convertType(out._internal,in);
}

template<typename T1,typename T2, typename std::enable_if<!isGridScalar<T1>::value, T1>::type* = nullptr>
accelerator_inline void convertType(T1 & out, const iScalar<T2> & in) {
  convertType(out,in._internal);
}

  template<typename T1,typename T2,int N>
accelerator_inline void convertType(iMatrix<T1,N> & out, const iMatrix<T2,N> & in) {
  for (int i=0;i<N;i++)
    for (int j=0;j<N;j++)
      convertType(out._internal[i][j],in._internal[i][j]);
}

template<typename T1,typename T2,int N>
accelerator_inline void convertType(iVector<T1,N> & out, const iVector<T2,N> & in) {
  for (int i=0;i<N;i++)
    convertType(out._internal[i],in._internal[i]);
}

template<typename T, typename std::enable_if<isGridFundamental<T>::value, T>::type* = nullptr>
accelerator_inline void convertType(T & out, const T & in) {
  out = in;
}

template<typename T1,typename T2>
accelerator_inline void convertType(Lattice<T1> & out, const Lattice<T2> & in) {
  auto out_v = out.View();
  auto in_v  = in.View();

  accelerator_for(ss,out_v.size(),T1::Nsimd(),{
      convertType(out_v[ss],in_v[ss]);
    });
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
      auto d2 = TensorRemove(innerProductD2(lhs_v(ss),rhs_v(ss)));
      iScalar<t_inner> d1;
      convertType(d1,d2);
      coalescedWrite(ret_v[ss],d1);
    });

  return ret;
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

  auto CoarseInner_  = CoarseInner.View();
  auto coarse_inner_ = coarse_inner.View();

  // Precision promotion
  fine_inner = localInnerProductD(fineX,fineY);
  cgpt_blockSum(coarse_inner,fine_inner);
  accelerator_for(ss, coarse->oSites(), 1, {
      convertType(CoarseInner_[ss], TensorRemove(coarse_inner_[ss]));
    });
 
}

template<class vobj,class vobj2,class CComplex>
  inline void cgpt_blockZAXPY(Lattice<vobj> &fineZ,
			      const Lattice<CComplex> &coarseA,
			      const Lattice<vobj2> &fineX,
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
      typename vobj2::tensor_reduced cA;
      vobj cAx;
      convertType(cA,TensorRemove(coarseA_(sc)));
      auto prod = cA*fineX_(sf);
      convertType(cAx,prod);
      coalescedWrite(fineZ_[sf],cAx+fineY_(sf));

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
    cgpt_blockZAXPY(fineDataRed,ip,*Basis[v],fineDataRed); 

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
      cgpt_blockZAXPY(*Basis[v],ip,*Basis[u],*Basis[v]);
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

  //Lattice<typename vobj::DoublePrecision2> fineDataD(fine);

  //fineDataD=Zero();
  fineData=Zero();
  for(int i=0;i<nbasis;i++) {
    Lattice<iScalar<CComplex> > ip = PeekIndex<0>(coarseData,i);
    auto  ip_ =  ip.View();
    cgpt_blockZAXPY(fineData,ip,*Basis[i],fineData);
    // cgpt_blockZAXPY(fineDataD,ip,*Basis[i],fineDataD);
  }

  //convertType(fineData,fineDataD);
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

