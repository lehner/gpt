/*
    GPT - Grid Python Toolkit
    Copyright (C) 2025  Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)


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

template<class Impl, class AslashedType>
class CayleyFermion5DWithVectorField : public CayleyFermion5D<Impl>
{
public:
  INHERIT_IMPL_TYPES(Impl);
  typedef typename FermionField::scalar_type   scalar_type;
  typedef typename FermionField::vector_type   vector_type;
public:

  Lattice<AslashedType> Aslashed, Aslashed_even, Aslashed_odd;
  deviceVector<iSpinMatrix<vector_type>> MooeeInv_matrices_even, MooeeInv_matrices_odd;
  
  /////////////////////////////////////////////////////
  // Instantiate different versions depending on Impl
  /////////////////////////////////////////////////////
  void M5D(const FermionField &psi_i,
	   const FermionField &phi_i,
	   FermionField &chi_i,
	   std::vector<Coeff_t> &lower,
	   std::vector<Coeff_t> &diag,
	   std::vector<Coeff_t> &upper) {
  
    chi_i.Checkerboard()=psi_i.Checkerboard();
    GridBase *grid=psi_i.Grid();
    autoView(psi , psi_i,AcceleratorRead);
    autoView(phi , phi_i,AcceleratorRead);
    autoView(chi , chi_i,AcceleratorWriteDiscard);
    autoView(As_full, Aslashed,AcceleratorRead);
    autoView(As_even, Aslashed_even,AcceleratorRead);
    autoView(As_odd, Aslashed_odd,AcceleratorRead);

    auto As = &As_full[0];
    if (grid->_isCheckerBoarded) {
      if (psi.Checkerboard() == Even) {
	As = &As_even[0];
      } else {
	As = &As_odd[0];
      }
    }
    assert(phi.Checkerboard() == psi.Checkerboard());
    
    int Ls =this->Ls;
    
    acceleratorCopyToDevice(&diag[0] ,&this->d_diag[0],Ls*sizeof(Coeff_t));
    acceleratorCopyToDevice(&upper[0],&this->d_upper[0],Ls*sizeof(Coeff_t));
    acceleratorCopyToDevice(&lower[0],&this->d_lower[0],Ls*sizeof(Coeff_t));
    
    auto pdiag = &this->d_diag[0];
    auto pupper = &this->d_upper[0];
    auto plower = &this->d_lower[0];

    double scale_A = upper[Ls-1] / this->mass_minus;
    assert( lower[0] / this->mass_plus == scale_A );

    // 10 = 3 complex mult + 2 complex add
    // Flops = 10.0*(Nc*Ns) *Ls*vol (/2 for red black counting) + Aslash
    uint64_t nloop = grid->oSites();
    accelerator_for(sss,nloop,Simd::Nsimd(),{
	uint64_t s = sss%Ls;
	uint64_t ss= sss-s;
	uint64_t x = sss/Ls;
	typedef decltype(coalescedRead(psi[0])) spinor;
	spinor tmp1, tmp2, tmp3, tmp4;
	uint64_t idx_u = ss+((s+1)%Ls);
	uint64_t idx_l = ss+((s+Ls-1)%Ls);
	spProj5m(tmp1,psi(idx_u));
	spProj5p(tmp2,psi(idx_l));
	spinor tmp5 = coalescedRead(As[x]) * psi(ss+s);
	spProj5m(tmp3,tmp5);
	spProj5p(tmp4,tmp5);
	double s4 = (s == 0) ? scale_A : 0.0;
	double s3 = (s == Ls - 1) ? scale_A : 0.0;
	coalescedWrite(chi[ss+s],pdiag[s]*phi(ss+s)+pupper[s]*tmp1+plower[s]*tmp2+s3*tmp3+s4*tmp4);
      });

  }

  void M5Ddag(const FermionField &psi_i,
	      const FermionField &phi_i,
	      FermionField &chi_i,
	      std::vector<Coeff_t> &lower,
	      std::vector<Coeff_t> &diag,
	      std::vector<Coeff_t> &upper) {
       chi_i.Checkerboard()=psi_i.Checkerboard();
       GridBase *grid=psi_i.Grid();
       autoView(psi , psi_i,AcceleratorRead);
       autoView(phi , phi_i,AcceleratorRead);
       autoView(chi , chi_i,AcceleratorWriteDiscard);
       autoView(As_full, Aslashed,AcceleratorRead);
       autoView(As_even, Aslashed_even,AcceleratorRead);
       autoView(As_odd, Aslashed_odd,AcceleratorRead);
       
       auto As = &As_full[0];
       if (grid->_isCheckerBoarded) {
	 if (psi.Checkerboard() == Even) {
	   As = &As_even[0];
	 } else {
	   As = &As_odd[0];
	 }
       }
       
       assert(phi.Checkerboard() == psi.Checkerboard());
 
       int Ls=this->Ls;
       
       acceleratorCopyToDevice(&diag[0] ,&this->d_diag[0],Ls*sizeof(Coeff_t));
       acceleratorCopyToDevice(&upper[0],&this->d_upper[0],Ls*sizeof(Coeff_t));
       acceleratorCopyToDevice(&lower[0],&this->d_lower[0],Ls*sizeof(Coeff_t));
       
       auto pdiag = &this->d_diag[0];
       auto pupper = &this->d_upper[0];
       auto plower = &this->d_lower[0];

       double scale_A = upper[Ls-1] / this->mass_plus;
       assert( lower[0] / this->mass_minus == scale_A );

       // Flops = 6.0*(Nc*Ns) *Ls*vol + Aslash
       uint64_t nloop = grid->oSites();
       accelerator_for(sss,nloop,Simd::Nsimd(),{
	   uint64_t s = sss%Ls;
	   uint64_t ss= sss-s;
	   uint64_t x = sss/Ls;
	   typedef decltype(coalescedRead(psi[0])) spinor;
	   spinor tmp1,tmp2,tmp3,tmp4;
	   uint64_t idx_u = ss+((s+1)%Ls);
	   uint64_t idx_l = ss+((s+Ls-1)%Ls);
	   spProj5p(tmp1,psi(idx_u));
	   spProj5m(tmp2,psi(idx_l));
	   spinor tmp5 = adj(coalescedRead(As[x])) * psi(ss+s);
	   spProj5p(tmp3,tmp5);
	   spProj5m(tmp4,tmp5);
	   double s4 = (s == 0) ? scale_A : 0.0;
	   double s3 = (s == Ls - 1) ? scale_A : 0.0;
	   coalescedWrite(chi[ss+s],pdiag[s]*phi(ss+s)+pupper[s]*tmp1+plower[s]*tmp2+s3*tmp3+s4*tmp4);
	 });

  }


  void MooeeInv(const FermionField& in, FermionField& out) {
    long oSites = in.Grid()->oSites();
    int Ls = this->Ls;
   
    autoView(v_o , out, AcceleratorWriteDiscard);
    autoView(v_i , in, AcceleratorRead);
    
    auto p_o = &v_o[0];
    auto p_i = &v_i[0];

    auto SM = (in.Checkerboard() == Even)
      ? (iSpinMatrix<vector_type>*)&MooeeInv_matrices_even[0]
      : (iSpinMatrix<vector_type>*)&MooeeInv_matrices_odd[0];
    
    accelerator_for(sss,oSites,Simd::Nsimd(),{
	long s  = sss % Ls;
	long os = sss / Ls;
	typedef decltype(coalescedRead(p_o[0])) spinor;
	spinor d = Zero();
	for (int sp=0;sp<Ls;sp++)
	  d += coalescedRead(SM[os * Ls * Ls + s*Ls + sp]) * coalescedRead(p_i[os * Ls + sp]);
	coalescedWrite(p_o[os * Ls + s], d);
      });
    
  }

  void MooeeInvDag(const FermionField& in, FermionField& out) {
    long oSites = in.Grid()->oSites();
    int Ls = this->Ls;
   
    autoView(v_o , out, AcceleratorWriteDiscard);
    autoView(v_i , in, AcceleratorRead);
    
    auto p_o = &v_o[0];
    auto p_i = &v_i[0];

    auto SM = (in.Checkerboard() == Even)
      ? (iSpinMatrix<vector_type>*)&MooeeInv_matrices_even[0]
      : (iSpinMatrix<vector_type>*)&MooeeInv_matrices_odd[0];
    
    accelerator_for(sss,oSites,Simd::Nsimd(),{
	long s  = sss % Ls;
	long os = sss / Ls;
	typedef decltype(coalescedRead(p_o[0])) spinor;
	spinor d = Zero();
	for (int sp=0;sp<Ls;sp++)
	  d += adj(coalescedRead(SM[os * Ls * Ls + sp*Ls + s])) * coalescedRead(p_i[os * Ls + sp]);
	coalescedWrite(p_o[os * Ls + s], d);
      });
    
  }

  void PrepareMeeooInv(int cb, deviceVector<iSpinMatrix<vector_type>>& SpinMatrices) {
    int Ls = this->Ls;
    long ldim = Ls * Ns;
    long sites = this->_FiveDimRedBlackGrid->_fsites;
    long sites4d = sites / Ls;
    long oSites = this->_FiveDimRedBlackGrid->oSites();
    deviceVector<scalar_type> matricesInv(ldim*ldim*sites4d);
    deviceVector<scalar_type> matrices(matricesInv.size());
    assert(SpinMatrices.size() == Ls*Ls*sites4d);
    FermionField tmp_i(this->_FiveDimRedBlackGrid), tmp_o(this->_FiveDimRedBlackGrid);
    
    for (int spin=0;spin<Ns;spin++) {
      for (int s=0;s<Ls;s++) {
	// populate
	{
	  autoView(v_tmp_i , tmp_i,AcceleratorWriteDiscard);
	  typedef decltype(coalescedRead(v_tmp_i[0])) spinor;
	  
	  spinor src = Zero();
	  src._internal._internal[spin]._internal[0] = 1;
	  
	  auto dst = &v_tmp_i[0];
	  accelerator_for(sss,oSites,Simd::Nsimd(),{
	      int _s = sss % Ls;
	      coalescedWrite(dst[sss], ((_s == s) ? (1.0) : (0.0)) * src);
	    });
	}
	// apply
	tmp_i.Checkerboard() = cb;
	tmp_o.Checkerboard() = cb;
	this->Mooee(tmp_i, tmp_o);
	
	// read off
	{
	  autoView(v_tmp_o , tmp_o,AcceleratorRead);
	  
	  auto dst = &matrices[0];
	  auto src = &v_tmp_o[0];
	  
	  accelerator_for(sss,oSites,1,{
	      long _s = sss % Ls;
	      long os = sss / Ls;
	      auto v = coalescedRead(src[sss]);
	      for (long ln=0;ln<Simd::Nsimd();ln++) {
		long ss = os * Simd::Nsimd() + ln;
		auto vs = extractLane(ln, v);
		for (int spinp=0;spinp<Ns;spinp++)
		  dst[ss*ldim*ldim + (_s*Ns + spinp)*ldim + (s*Ns + spin)] = vs._internal._internal[spinp]._internal[0];
	      }
	    });
	}
      }
    }

    // next, create inverse with blas
    {
      deviceVector<scalar_type*> pmatrices(sites4d);
      deviceVector<scalar_type*> pmatricesInv(sites4d);
      auto psrc = &pmatrices[0];
      auto pdst = &pmatricesInv[0];
      auto src = &matrices[0];
      auto dst = &matricesInv[0];
      accelerator_for(ss,sites4d,1,{ psrc[ss] = &src[ss*ldim*ldim]; pdst[ss] = &dst[ss*ldim*ldim]; });
      GridBLAS blas;
      blas.inverseBatched((int64_t)ldim, pmatrices, pmatricesInv);
    }

    // now re-introduce Grid data layout
    {
      auto dst = (iSpinMatrix<vector_type>*)&SpinMatrices[0];
      auto src = &matricesInv[0];
      
      accelerator_for(sss,oSites,1,{
	long s  = sss % Ls;
	long os = sss / Ls;

	for (long ln=0;ln<Simd::Nsimd();ln++) {
	  long ss = os * Simd::Nsimd() + ln;

	  for (int sp=0;sp<Ls;sp++) {
	    iSpinMatrix<scalar_type> sm;
	    for (int spin=0;spin<Ns;spin++)
	      for (int spinp=0;spinp<Ns;spinp++)
		sm._internal._internal[spinp][spin] = src[ss*ldim*ldim + (sp*Ns + spinp)*ldim + (s*Ns + spin)];

	    insertLane(ln, dst[os*Ls*Ls + sp*Ls + s], sm);
	  }
	}
      });
    }

    // should do a test
    //GridParallelRNG          pRNG(this->_FiveDimGrid);      pRNG.SeedFixedIntegers(std::vector<int>({45,12,81,9}));
    //FermionField tmp_i_full(this->_FiveDimGrid);
    //random(pRNG,tmp_i_full);
    //pickCheckerboard(tmp_i.Checkerboard(),tmp_i,tmp_i_full);
    //tmp_o.Checkerboard() = tmp_i.Checkerboard();
    //this->Mooee(tmp_i, tmp_o);

    // apply MooeeInv
    //FermionField tmp2_o(this->_FiveDimRedBlackGrid);
    //tmp2_o.Checkerboard() = tmp_i.Checkerboard();
    //this->MooeeInv(tmp_o, tmp2_o);
    //RealD err = norm2(closure(tmp_i - tmp2_o)) / norm2(tmp_i);
    //std::cout << GridLogMessage << "Error: " << err << std::endl ;
  }
  
  void UpdateAslashed() {

    pickCheckerboard(Even,Aslashed_even,Aslashed);
    pickCheckerboard(Odd,Aslashed_odd,Aslashed);

    // in presence of Aslashed field, have to perform explicit Mooee inversion
    // matrix should be diagonal in color space but non-trivialin Ls x spin space
    int Ls = this->Ls;
    long ldim = Ls * Ns;
    long sites4d = this->_FiveDimRedBlackGrid->_fsites / Ls;

    PrepareMeeooInv(Even, MooeeInv_matrices_even);
    PrepareMeeooInv(Odd, MooeeInv_matrices_odd);
  }

  void ImportGauge(const GaugeField &_Umu) {
    CayleyFermion5D<Impl>::ImportGauge(_Umu);
    UpdateAslashed();
  }
  
  ///////////////////////////////////////////////////////////////
  // Constructors
  ///////////////////////////////////////////////////////////////
  CayleyFermion5DWithVectorField(GaugeField &_Umu,
				 Lattice<AslashedType> &_Aslashed,
				 GridCartesian         &FiveDimGrid,
				 GridRedBlackCartesian &FiveDimRedBlackGrid,
				 GridCartesian         &FourDimGrid,
				 GridRedBlackCartesian &FourDimRedBlackGrid,
				 RealD _mass,RealD _M5,const ImplParams &p= ImplParams()) :
    CayleyFermion5D<Impl>(_Umu,FiveDimGrid,FiveDimRedBlackGrid,FourDimGrid,FourDimRedBlackGrid,_mass,_M5,p),
    Aslashed(_Aslashed), Aslashed_even(&FourDimRedBlackGrid), Aslashed_odd(&FourDimRedBlackGrid) {

    MooeeInv_matrices_even.resize(FiveDimRedBlackGrid._fsites * this->Ls);
    MooeeInv_matrices_odd.resize(FiveDimRedBlackGrid._fsites * this->Ls);

    /*
    D- = c DW - 1
    D+ = b DW + 1

    D+ (s<-s) + D- P+ (s<-s-1) + D- P- (s<-s+1) - m (D- P+ (1<-Ls) + D- P- (Ls<-1))
    + i e D- P- Aslashed (Ls<-Ls) + i e D- P+ Aslashed (1<-1)

    Input to DW should be:
    b (s<-s) + c P+ (s<-s-1) + c P- (s<-s+1) - m c (P+ (1<-Ls) + P- (Ls<-1)) + ie c P- Aslashed (Ls<-Ls) + i e c P+ Aslashed (1<-1)

    // check against Meooe5D
    From input to output without DW should be:
    1 (s<-s)  - P+ (s<-s-1) - P- (s<-s+1) + m (P+ (1<-Ls) + P- (Ls<-1)) - ie P- Aslashed (Ls<-Ls) -ie P+ Aslashed (1<-1)

   */
  }
};

template<class Impl, class AslashedType>
class MobiusFermionWithVectorField : public CayleyFermion5DWithVectorField<Impl, AslashedType>
{
public:
  INHERIT_IMPL_TYPES(Impl);
public:
  virtual void Instantiatable() { };

  // Constructors
  MobiusFermionWithVectorField(GaugeField &_Umu, Lattice<AslashedType>& _Aslashed,
			       GridCartesian         &FiveDimGrid,
			       GridRedBlackCartesian &FiveDimRedBlackGrid,
			       GridCartesian         &FourDimGrid,
			       GridRedBlackCartesian &FourDimRedBlackGrid,
			       RealD _mass,RealD _M5,
			       RealD b, RealD c,const ImplParams &p= ImplParams()) : 
      
    CayleyFermion5DWithVectorField<Impl, AslashedType>(_Umu, _Aslashed,
						       FiveDimGrid,
						       FiveDimRedBlackGrid,
						       FourDimGrid,
						       FourDimRedBlackGrid,_mass,_M5,p)
    
  {
    RealD eps = 1.0;
    Approx::zolotarev_data *zdata = Approx::higham(eps,this->Ls);// eps is ignored for higham
    assert(zdata->n==this->Ls);
    this->SetCoefficientsTanh(zdata,b,c);
    Approx::zolotarev_free(zdata);
  }

  void SetCoefficientsInternal(RealD zolo_hi,std::vector<Coeff_t> & gamma,RealD b,RealD c) {
    CayleyFermion5DWithVectorField<Impl, AslashedType>::SetCoefficientsInternal(zolo_hi, gamma,b,c);
    this->UpdateAslashed();
  }

};

