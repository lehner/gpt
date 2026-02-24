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


#if 1
  typedef iSpinMatrix<scalar_type> spin_matrix_type;

  spin_matrix_type spin_inverse(const spin_matrix_type& in) {
    spin_matrix_type out;
    Eigen::MatrixXcd e_in = Eigen::MatrixXcd::Zero(Ns,Ns);
    for (int a=0;a<Ns;a++)
      for (int b=0;b<Ns;b++)
	e_in(a,b) = in._internal._internal[a][b]._internal;
    Eigen::MatrixXcd e_out = e_in.inverse();
    for (int a=0;a<Ns;a++)
      for (int b=0;b<Ns;b++)
	out._internal._internal[a][b]._internal = e_out(a,b);
    return out;
  }
  
  void MooeeInv(const FermionField& psi_i, FermionField& chi_i) {
    chi_i.Checkerboard()=psi_i.Checkerboard();
    GridBase *grid=psi_i.Grid();
    
    autoView(psi , psi_i,AcceleratorRead);
    autoView(chi , chi_i,AcceleratorWriteDiscard);
    
    int Ls=this->Ls;

    std::vector<spin_matrix_type> lee, dee, inv_dee, leem, uee, ueem;
    deviceVector<spin_matrix_type> d_lee, d_dee, d_inv_dee, d_uee, d_leem, d_ueem;

    // next step: bee needs to be a spin field, therefore ALL components need to be spin fields

    std::vector<spin_matrix_type> bee(Ls), inv_bee(Ls);
    for (int i=0;i<Ls;i++) {
      bee[i] = this->bee[i];
      //if (i == 0) {
      //	bee[i] += Gamma::Gamma5 * Aslashed
      //}
      inv_bee[i] = spin_inverse(bee[i]);
    }

    dee.resize(Ls);
    inv_dee.resize(Ls);
    lee.resize(Ls);
    leem.resize(Ls);
    uee.resize(Ls);
    ueem.resize(Ls);

    for(int i=0;i<Ls;i++){
      
      dee[i] = bee[i];
      
      if ( i < Ls-1 ) {

	lee[i] =-this->cee[i+1]*inv_bee[i]; // sub-diag entry on the ith column
	
	leem[i]=this->mass_minus*this->cee[Ls-1]*inv_bee[0];
	for(int j=0;j<i;j++) {
	  leem[i]*= this->aee[j]*inv_bee[j+1];
	}
	
	uee[i] =-this->aee[i]*inv_bee[i];   // up-diag entry on the ith row
	
	ueem[i]=this->mass_plus;
	for(int j=1;j<=i;j++) ueem[i]*= this->cee[j]*inv_bee[j];
	ueem[i]*= this->aee[0]*inv_bee[0];
	
      } else { 
	lee[i] =0.0;
	leem[i]=0.0;
	uee[i] =0.0;
	ueem[i]=0.0;
      }
    }
    
    { 
      spin_matrix_type delta_d=(spin_matrix_type)this->mass_minus*this->cee[Ls-1];
      for(int j=0;j<Ls-1;j++) {
	delta_d *= this->cee[j]*inv_bee[j];
      }
      dee[Ls-1] += delta_d;
    }

    for (int i=0;i<Ls;i++)
      inv_dee[i] = spin_inverse(dee[i]);

    std::cout << GridLogMessage << "Test2" << std::endl;

    d_dee.resize(Ls);
    d_inv_dee.resize(Ls);
    d_lee.resize(Ls);
    d_uee.resize(Ls);
    d_leem.resize(Ls);
    d_ueem.resize(Ls);
    
    acceleratorCopyToDevice(&lee[0],&d_lee[0],Ls*sizeof(lee[0]));
    acceleratorCopyToDevice(&dee[0],&d_dee[0],Ls*sizeof(lee[0]));
    acceleratorCopyToDevice(&inv_dee[0],&d_inv_dee[0],Ls*sizeof(lee[0]));
    acceleratorCopyToDevice(&uee[0],&d_uee[0],Ls*sizeof(lee[0]));
    acceleratorCopyToDevice(&leem[0],&d_leem[0],Ls*sizeof(lee[0]));
    acceleratorCopyToDevice(&ueem[0],&d_ueem[0],Ls*sizeof(lee[0]));
    
    auto plee  = & d_lee [0];
    auto pdee  = & d_dee [0];
    auto p_inv_dee  = & d_inv_dee [0];
    auto puee  = & d_uee [0];
    auto pleem = & d_leem[0];
    auto pueem = & d_ueem[0];
    
    uint64_t nloop = grid->oSites()/Ls;
    accelerator_for(sss,nloop,Simd::Nsimd(),{
	uint64_t ss=sss*Ls;
	typedef decltype(coalescedRead(psi[0])) spinor;
	spinor tmp, acc, res;
	
	// X = Nc*Ns
	// flops = 2X + (Ls-2)(4X + 4X) + 6X + 1 + 2X + (Ls-1)(10X + 1) = -16X + Ls(1+18X) = -192 + 217*Ls flops
	// Apply (L^{\prime})^{-1} L_m^{-1}
	res = psi(ss);
	spProj5m(tmp,res);
	acc = pleem[0]*tmp;
	spProj5p(tmp,res);
	coalescedWrite(chi[ss],res);
	
	for(int s=1;s<Ls-1;s++){
	  res = psi(ss+s);
	  res -= plee[s-1]*tmp;
	  spProj5m(tmp,res);
	  acc += pleem[s]*tmp;
	  spProj5p(tmp,res);
	  coalescedWrite(chi[ss+s],res);
	}
	res = psi(ss+Ls-1) - plee[Ls-2]*tmp - acc;
	
	// Apply U_m^{-1} D^{-1} U^{-1}
	res = (p_inv_dee[Ls-1])*res;
	coalescedWrite(chi[ss+Ls-1],res);
	spProj5p(acc,res);
	spProj5m(tmp,res);
	for (int s=Ls-2;s>=0;s--){
	  res = (p_inv_dee[s])*chi(ss+s) - puee[s]*tmp - pueem[s]*acc;
	  spProj5m(tmp,res);
	  coalescedWrite(chi[ss+s],res);
	}
      });

#else
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
#endif    
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
	  
	  auto dst = &v_tmp_i[0];
	  accelerator_for(sss,oSites,Simd::Nsimd(),{
	      int _s = sss % Ls;

	      spinor src = Zero();
	      src._internal._internal[spin]._internal[0] = 1;
	  
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
	      for (long ln=0;ln<Simd::Nsimd();ln++) {
		long ss = os * Simd::Nsimd() + ln;
		auto vs = extractLane(ln, src[sss]);
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

