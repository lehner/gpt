/*
    GPT - Grid Python Toolkit
    Copyright (C) 2020  Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)
                  2020  Daniel Richtmann (daniel.richtmann@ur.de, https://github.com/lehner/gpt)

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



////////////////////////////////////////////////////////////////////////////////////////////
// cgpt_convertType
////////////////////////////////////////////////////////////////////////////////////////////
accelerator_inline void cgpt_convertType(ComplexD & out, const std::complex<double> & in) {
  out = in;
}

accelerator_inline void cgpt_convertType(ComplexF & out, const std::complex<float> & in) {
  out = in;
}

template<typename T>
accelerator_inline EnableIf<isGridFundamental<T>> cgpt_convertType(T & out, const T & in) {
  out = in;
}

#ifdef GRID_SIMT
accelerator_inline void cgpt_convertType(vComplexF & out, const ComplexF & in) {
  ((ComplexF*)&out)[acceleratorSIMTlane(vComplexF::Nsimd())] = in;
}
accelerator_inline void cgpt_convertType(vComplexF & out, const ComplexD & in) {
  ((ComplexF*)&out)[acceleratorSIMTlane(vComplexF::Nsimd())] = in;
}
accelerator_inline void cgpt_convertType(vComplexD & out, const ComplexD & in) {
  ((ComplexD*)&out)[acceleratorSIMTlane(vComplexD::Nsimd())] = in;
}
accelerator_inline void cgpt_convertType(vComplexD2 & out, const ComplexD & in) {
  ((ComplexD*)&out)[acceleratorSIMTlane(vComplexD::Nsimd()*2)] = in;
}
#endif

accelerator_inline void cgpt_convertType(vComplexF & out, const vComplexD2 & in) {
  //out.v = Optimization::PrecisionChange::DtoS(in.v[0],in.v[1]);
  precisionChange(out,in);
}

accelerator_inline void cgpt_convertType(vComplexD2 & out, const vComplexF & in) {
  //Optimization::PrecisionChange::StoD(in.v,out.v[0],out.v[1]);
  precisionChange(out,in);
}

template<typename T1,typename T2>
accelerator_inline void cgpt_convertType(iScalar<T1> & out, const iScalar<T2> & in) {
  cgpt_convertType(out._internal,in._internal);
}

template<typename T1,typename T2>
accelerator_inline NotEnableIf<isGridScalar<T1>> cgpt_convertType(T1 & out, const iScalar<T2> & in) {
  cgpt_convertType(out,in._internal);
}

template<typename T1,typename T2>
accelerator_inline NotEnableIf<isGridScalar<T2>> cgpt_convertType(iScalar<T1> & out, const T2 & in) {
  cgpt_convertType(out._internal,in);
}

template<typename T1,typename T2,int N>
accelerator_inline void cgpt_convertType(iMatrix<T1,N> & out, const iMatrix<T2,N> & in) {
  for (int i=0;i<N;i++)
    for (int j=0;j<N;j++)
      cgpt_convertType(out._internal[i][j],in._internal[i][j]);
}

template<typename T1,typename T2,int N>
accelerator_inline void cgpt_convertType(iVector<T1,N> & out, const iVector<T2,N> & in) {
  for (int i=0;i<N;i++)
    cgpt_convertType(out._internal[i],in._internal[i]);
}

template<typename T1,typename T2>
accelerator_inline void cgpt_convertType(Lattice<T1> & out, const Lattice<T2> & in) {
  autoView( out_v , out,AcceleratorWrite);
  autoView( in_v  , in ,AcceleratorRead);
  accelerator_for(ss,out_v.size(),T1::Nsimd(),{
      cgpt_convertType(out_v[ss],in_v(ss));
  });
}

////////////////////////////////////////////////////////////////////////////////////////////



template<class CComplex,class VLattice>
inline void vectorBlockOrthonormalize(Lattice<CComplex> &ip,VLattice &Basis, size_t n_virtual)
{
  GridBase *coarse = ip.Grid();
  GridBase *fine   = Basis[0].Grid();

  assert(Basis.size() % n_virtual == 0);
  size_t    nbasis = Basis.size() / n_virtual;

  // checks
  subdivides(coarse,fine);
  for (size_t j=0;j<Basis.size();j++)
    conformable(Basis[j].Grid(),fine);

  Lattice<CComplex> sip(coarse);
  typename std::remove_reference<decltype(Basis[0])>::type zz(fine);
  zz = Zero();
  zz.Checkerboard()=Basis[0].Checkerboard();
  for(size_t v=0;v<nbasis;v++) {
    for(size_t u=0;u<v;u++) {
      //Inner product & remove component
      sip = Zero();
      for (size_t j=0;j<n_virtual;j++) {
	blockInnerProductD(ip,Basis[j + u*n_virtual],Basis[j + v*n_virtual]);
	sip -= ip;
      }
      for (size_t j=0;j<n_virtual;j++)
	blockZAXPY(Basis[j + v*n_virtual],sip,Basis[j + u*n_virtual],Basis[j + v*n_virtual]);
    }

    // block normalize
    sip = Zero();
    for (size_t j=0;j<n_virtual;j++) {
      blockInnerProductD(ip,Basis[j + v*n_virtual],Basis[j + v*n_virtual]);
      sip += ip;
    }
    sip = pow(sip,-0.5);
    ip = Zero();
    for (size_t j=0;j<n_virtual;j++)
      blockZAXPY(Basis[j + v*n_virtual],sip,Basis[j + v*n_virtual],zz);
  }
}


template<class vobj,class CComplex,int basis_virtual_size,class VLattice,class T_singlet>
inline void vectorizableBlockProject(PVector<Lattice<iVector<CComplex, basis_virtual_size>>>&   coarse,
				     long                                                       coarse_n_virtual,
				     const PVector<Lattice<vobj>>&                              fine,
				     long                                                       fine_n_virtual,
				     const VLattice&                                            basis,
				     long                                                       basis_n_virtual,
				     const cgpt_block_lookup_table<T_singlet>&                  lut,
				     long                                                       basis_n_block)
{

  assert(fine.size() > 0 && coarse.size() > 0 && basis.size() > 0);

  assert(basis.size() % basis_n_virtual == 0);
  long basis_n = basis.size() / basis_n_virtual;

  assert(fine.size() % fine_n_virtual == 0);
  long fine_n = fine.size() / fine_n_virtual;

  assert(coarse.size() % coarse_n_virtual == 0);
  long coarse_n = coarse.size() / coarse_n_virtual;

  assert(fine_n == coarse_n);
  long vec_n = fine_n;

  assert(basis_n % coarse_n_virtual == 0);
  long coarse_virtual_size = basis_n / coarse_n_virtual;

  GridBase *fine_grid   = fine[0].Grid();
  GridBase *coarse_grid = coarse[0].Grid();

  long coarse_osites = coarse_grid->oSites();

  assert(fine_grid->_ndimension == coarse_grid->_ndimension);
  assert(lut.gridsMatch(coarse_grid, fine_grid));

  assert(fine_n_virtual == basis_n_virtual);

  auto lut_v = lut.View();
  auto sizes_v = lut.Sizes();

  VECTOR_VIEW_OPEN(fine,fine_v,AcceleratorRead);
  VECTOR_VIEW_OPEN(coarse,coarse_v,AcceleratorWriteDiscard);

  for (long basis_i0=0;basis_i0<basis_n;basis_i0+=basis_n_block) {
    long basis_i1 = std::min(basis_i0 + basis_n_block, basis_n);
    long basis_block = basis_i1 - basis_i0;
    VECTOR_VIEW_OPEN(basis.slice(basis_i0*fine_n_virtual,basis_i1*fine_n_virtual),basis_v,AcceleratorRead);

    accelerator_for(_idx, basis_block*coarse_osites*vec_n, vobj::Nsimd(), {
	auto idx = _idx;
	auto basis_i_rel = idx % basis_block; idx /= basis_block;
	auto basis_i_abs = basis_i_rel + basis_i0;
	auto vec_i = idx % vec_n; idx /= vec_n;
	auto sc = idx % coarse_osites; idx /= coarse_osites;
	
	decltype(innerProductD2(coalescedRead(basis_v[0][0]), coalescedRead(fine_v[0][0]))) reduce = Zero();
	
	for (long fine_virtual_i=0; fine_virtual_i<fine_n_virtual; fine_virtual_i++) {
	  for(long j=0; j<sizes_v[sc]; ++j) {
	    long sf = lut_v[sc][j];
	    reduce = reduce + innerProductD2(coalescedRead(basis_v[basis_i_rel*fine_n_virtual + fine_virtual_i][sf]),
					     coalescedRead(fine_v[vec_i*fine_n_virtual + fine_virtual_i][sf]));
	  }
	}
	
	long coarse_virtual_i = basis_i_abs / coarse_virtual_size;
	long coarse_i = basis_i_abs % coarse_virtual_size;
	cgpt_convertType(coarse_v[vec_i*coarse_n_virtual + coarse_virtual_i][sc](coarse_i), TensorRemove(reduce));
      });

    VECTOR_VIEW_CLOSE(basis_v);
  }
  
  VECTOR_VIEW_CLOSE(fine_v);
  VECTOR_VIEW_CLOSE(coarse_v);
}

template<class vobj,class CComplex,int basis_virtual_size,class VLattice,class T_singlet>
inline void vectorizableBlockPromote(PVector<Lattice<iVector<CComplex, basis_virtual_size>>>&   coarse,
				     long                                                       coarse_n_virtual,
				     const PVector<Lattice<vobj>>&                              fine,
				     long                                                       fine_n_virtual,
				     const VLattice&                                            basis,
				     long                                                       basis_n_virtual,
				     const cgpt_block_lookup_table<T_singlet>&                  lut,
				     long                                                       basis_n_block)
{

  assert(fine.size() > 0 && coarse.size() > 0 && basis.size() > 0);

  assert(basis.size() % basis_n_virtual == 0);
  long basis_n = basis.size() / basis_n_virtual;

  assert(fine.size() % fine_n_virtual == 0);
  long fine_n = fine.size() / fine_n_virtual;

  assert(coarse.size() % coarse_n_virtual == 0);
  long coarse_n = coarse.size() / coarse_n_virtual;

  assert(fine_n == coarse_n);
  long vec_n = fine_n;

  assert(basis_n % coarse_n_virtual == 0);
  long coarse_virtual_size = basis_n / coarse_n_virtual;

  GridBase *fine_grid   = fine[0].Grid();
  GridBase *coarse_grid = coarse[0].Grid();

  long fine_osites = fine_grid->oSites();

  assert(fine_grid->_ndimension == coarse_grid->_ndimension);
  assert(lut.gridsMatch(coarse_grid, fine_grid));

  assert(fine_n_virtual == basis_n_virtual);

  auto rlut_v = lut.ReverseView();

  VECTOR_VIEW_OPEN(fine,fine_v,AcceleratorWriteDiscard);
  VECTOR_VIEW_OPEN(coarse,coarse_v,AcceleratorRead);

  for (long basis_i0=0;basis_i0<basis_n;basis_i0+=basis_n_block) {
    long basis_i1 = std::min(basis_i0 + basis_n_block, basis_n);
    long basis_block = basis_i1 - basis_i0;
    VECTOR_VIEW_OPEN(basis.slice(basis_i0*fine_n_virtual,basis_i1*fine_n_virtual),basis_v,AcceleratorRead);

    accelerator_for(_idx, fine_osites*vec_n, vobj::Nsimd(), {
	
	auto idx = _idx;
	auto vec_i = idx % vec_n; idx /= vec_n;
	auto sf = idx % fine_osites; idx /= fine_osites;
	auto sc = rlut_v[sf];
	
#ifdef GRID_SIMT
	typename vobj::tensor_reduced::scalar_object cA;
	typename vobj::scalar_object cAx;
#else
	typename vobj::tensor_reduced cA;
	vobj cAx;
#endif
	
	for (long fine_virtual_i=0; fine_virtual_i<fine_n_virtual; fine_virtual_i++) {
	  decltype(cAx) fine_t;
	  if (basis_i0 == 0)
	    fine_t = Zero();
	  else
	    fine_t = coalescedRead(fine_v[vec_i*fine_n_virtual + fine_virtual_i][sf]);

	  for(long basis_i_rel=0; basis_i_rel<basis_block; basis_i_rel++) {
	    long basis_i_abs = basis_i_rel + basis_i0;
	    long coarse_virtual_i = basis_i_abs / coarse_virtual_size;
	    long coarse_i = basis_i_abs % coarse_virtual_size;
	    cgpt_convertType(cA,TensorRemove(coalescedRead(coarse_v[vec_i*coarse_n_virtual + coarse_virtual_i][sc])(coarse_i)));
	    auto prod = cA*coalescedRead(basis_v[basis_i_rel*fine_n_virtual + fine_virtual_i][sf]);
	    cgpt_convertType(cAx,prod);
	    fine_t = fine_t + cAx;
	  }
	
	  coalescedWrite(fine_v[vec_i*fine_n_virtual + fine_virtual_i][sf], fine_t);
	}
      });

    VECTOR_VIEW_CLOSE(basis_v);
  }
  
  VECTOR_VIEW_CLOSE(fine_v);
  VECTOR_VIEW_CLOSE(coarse_v);
}

template<class vobj,class T_singlet>
inline void vectorizableBlockSum(PVector<Lattice<vobj>>&                                    coarse,
				 long                                                       coarse_n_virtual,
				 const PVector<Lattice<vobj>>&                              fine,
				 long                                                       fine_n_virtual,
				 const cgpt_block_lookup_table<T_singlet>&                  lut)
{

  assert(fine.size() > 0 && coarse.size() > 0);

  assert(fine.size() % fine_n_virtual == 0);
  long fine_n = fine.size() / fine_n_virtual;

  assert(coarse.size() % coarse_n_virtual == 0);
  long coarse_n = coarse.size() / coarse_n_virtual;

  assert(fine_n == coarse_n);
  long vec_n = fine_n;

  assert(coarse_n_virtual == fine_n_virtual);

  GridBase *fine_grid   = fine[0].Grid();
  GridBase *coarse_grid = coarse[0].Grid();

  long coarse_osites = coarse_grid->oSites();

  assert(fine_grid->_ndimension == coarse_grid->_ndimension);
  assert(lut.gridsMatch(coarse_grid, fine_grid));

  auto lut_v = lut.View();
  auto sizes_v = lut.Sizes();

  VECTOR_VIEW_OPEN(fine,fine_v,AcceleratorRead);
  VECTOR_VIEW_OPEN(coarse,coarse_v,AcceleratorWriteDiscard);

#ifdef GRID_SIMT
  typedef typename vobj::scalar_object reduce_obj;
#else
  typedef vobj reduce_obj;
#endif

  accelerator_for(_idx, coarse_osites*vec_n, vobj::Nsimd(), {
      auto idx = _idx;
      auto vec_i = idx % vec_n; idx /= vec_n;
      auto sc = idx % coarse_osites; idx /= coarse_osites;
   
      for (long virtual_i=0; virtual_i<fine_n_virtual; virtual_i++) {

	reduce_obj reduce = Zero();
	
	for(long j=0; j<sizes_v[sc]; ++j) {
	  long sf = lut_v[sc][j];
	  reduce = reduce + coalescedRead(fine_v[vec_i*fine_n_virtual + virtual_i][sf]);
	}

	coalescedWrite(coarse_v[vec_i*coarse_n_virtual + virtual_i][sc], reduce);
      }
    });
  
  VECTOR_VIEW_CLOSE(fine_v);
  VECTOR_VIEW_CLOSE(coarse_v);
}


template<class vobj,class T_singlet>
inline void vectorizableBlockEmbed(PVector<Lattice<vobj>>&                                    coarse,
				   long                                                       coarse_n_virtual,
				   const PVector<Lattice<vobj>>&                              fine,
				   long                                                       fine_n_virtual,
				   const cgpt_block_lookup_table<T_singlet>&                  lut)
{

  assert(fine.size() > 0 && coarse.size() > 0);

  assert(fine.size() % fine_n_virtual == 0);
  long fine_n = fine.size() / fine_n_virtual;

  assert(coarse.size() % coarse_n_virtual == 0);
  long coarse_n = coarse.size() / coarse_n_virtual;

  assert(fine_n == coarse_n);
  long vec_n = fine_n;

  assert(coarse_n_virtual == fine_n_virtual);

  GridBase *fine_grid   = fine[0].Grid();
  GridBase *coarse_grid = coarse[0].Grid();

  long coarse_osites = coarse_grid->oSites();

  assert(fine_grid->_ndimension == coarse_grid->_ndimension);
  assert(lut.gridsMatch(coarse_grid, fine_grid));

  auto lut_v = lut.View();
  auto sizes_v = lut.Sizes();

  VECTOR_VIEW_OPEN(fine,fine_v,AcceleratorWriteDiscard);
  VECTOR_VIEW_OPEN(coarse,coarse_v,AcceleratorRead);

  accelerator_for(_idx, coarse_osites*vec_n, vobj::Nsimd(), {
      auto idx = _idx;
      auto vec_i = idx % vec_n; idx /= vec_n;
      auto sc = idx % coarse_osites; idx /= coarse_osites;
   
      for (long virtual_i=0; virtual_i<fine_n_virtual; virtual_i++) {

	auto val = coalescedRead(coarse_v[vec_i*coarse_n_virtual + virtual_i][sc]);
	
	for(long j=0; j<sizes_v[sc]; ++j) {
	  long sf = lut_v[sc][j];
	  coalescedWrite(fine_v[vec_i*fine_n_virtual + virtual_i][sf], val);
	}
      }
    });
  
  VECTOR_VIEW_CLOSE(fine_v);
  VECTOR_VIEW_CLOSE(coarse_v);
}
