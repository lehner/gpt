/*
  CGPT

  Authors: Christoph Lehner 2020
*/
#include "lib.h"

// need to be able to handle pointers to play nicely with numpy views
void cgpt_QR_decomp(RealD* lmd,                // Nm
		    RealD* lme,                // Nm 
		    int Nk, int Nm,            // Nk, Nm
		    RealD* Qt,                 // Nm x Nm matrix
		    RealD Dsh, int kmin, int kmax) {

  int k = kmin-1;
  RealD x;
    
  RealD Fden = 1.0/hypot(lmd[k]-Dsh,lme[k]);
  RealD c = ( lmd[k] -Dsh) *Fden;
  RealD s = -lme[k] *Fden;
      
  RealD tmpa1 = lmd[k];
  RealD tmpa2 = lmd[k+1];
  RealD tmpb  = lme[k];

  lmd[k]   = c*c*tmpa1 +s*s*tmpa2 -2.0*c*s*tmpb;
  lmd[k+1] = s*s*tmpa1 +c*c*tmpa2 +2.0*c*s*tmpb;
  lme[k]   = c*s*(tmpa1-tmpa2) +(c*c-s*s)*tmpb;
  x        =-s*lme[k+1];
  lme[k+1] = c*lme[k+1];
      
  for(int i=0; i<Nk; ++i){
    RealD Qtmp1    = Qt[k*Nm+i];
    RealD Qtmp2    = Qt[(k+1)*Nm+i];
    Qt[k*Nm+i]     = c*Qtmp1 - s*Qtmp2;
    Qt[(k+1)*Nm+i]= s*Qtmp1 + c*Qtmp2; 
  }

  // Givens transformations
  for(int k = kmin; k < kmax-1; ++k){
      
    RealD Fden = 1.0/hypot(x,lme[k-1]);
    RealD c = lme[k-1]*Fden;
    RealD s = - x*Fden;
    
    RealD tmpa1 = lmd[k];
    RealD tmpa2 = lmd[k+1];
    RealD tmpb  = lme[k];

    lmd[k]   = c*c*tmpa1 +s*s*tmpa2 -2.0*c*s*tmpb;
    lmd[k+1] = s*s*tmpa1 +c*c*tmpa2 +2.0*c*s*tmpb;
    lme[k]   = c*s*(tmpa1-tmpa2) +(c*c-s*s)*tmpb;
    lme[k-1] = c*lme[k-1] -s*x;

    if(k != kmax-2){
      x = -s*lme[k+1];
      lme[k+1] = c*lme[k+1];
    }

    for(int i=0; i<Nk; ++i){
      RealD Qtmp1    = Qt[k*Nm+i];
      RealD Qtmp2    = Qt[(k+1)*Nm+i];
      Qt[k*Nm+i]     = c*Qtmp1 -s*Qtmp2;
      Qt[(k+1)*Nm+i] = s*Qtmp1 +c*Qtmp2;
    }
  }
}

EXPORT(qr_decomp,{

    PyObject* _lmd,* _lme,* _Qt,* _Dsh;
    int Nk,Nm,kmin,kmax;
    if (!PyArg_ParseTuple(args, "OOiiOOii", &_lmd, &_lme, &Nk, &Nm, &_Qt, &_Dsh, &kmin, &kmax)) {
      return NULL;
    }

    RealD Dsh;
    cgpt_convert(_Dsh,Dsh);

    RealD* Qt;
    int NQt;
    cgpt_numpy_import_matrix(_Qt,Qt,NQt);

    RealD* lmd;
    int Nlmd;
    cgpt_numpy_import_vector(_lmd,lmd,Nlmd);

    RealD* lme;
    int Nlme;
    cgpt_numpy_import_vector(_lme,lme,Nlme);

    ASSERT(Nlme == Nm && Nlmd == Nm && NQt == Nm && Nk <= Nm);
    cgpt_QR_decomp(lmd,lme,Nk,Nm,Qt,Dsh,kmin,kmax);

    return PyLong_FromLong(0);
  });

EXPORT(rotate,{

    PyObject* _basis,* _Qt;
    int j0,j1,k0,k1;
    if (!PyArg_ParseTuple(args, "OOiiii", &_basis, &_Qt, &j0, &j1, &k0, &k1)) {
      return NULL;
    }
    
    std::vector<cgpt_Lattice_base*> basis;
    cgpt_basis_fill(basis,_basis);

    ASSERT(basis.size() > 0);

    RealD* data;
    int Nm;
    cgpt_numpy_import_matrix(_Qt,data,Nm);

    ASSERT(j0 <= j1 && k0 <= k1 && j0 >=0 && k0 >= 0 && k1 <= Nm && j1 <= Nm && 
	   (int)basis.size() >= j1 && (int)basis.size() >= k1);

    basis[0]->basis_rotate(basis,data,j0,j1,k0,k1,Nm);
    
    return PyLong_FromLong(0);
  });

EXPORT(linear_combination,{

    PyObject* _basis,* _Qt;
    void* _dst;
    if (!PyArg_ParseTuple(args, "lOO", &_dst, &_basis, &_Qt)) {
      return NULL;
    }
    
    std::vector<cgpt_Lattice_base*> basis;
    cgpt_basis_fill(basis,_basis);

    cgpt_Lattice_base* dst = (cgpt_Lattice_base*)_dst;

    ASSERT(basis.size() > 0);

    RealD* data;
    int Nm;
    cgpt_numpy_import_vector(_Qt,data,Nm);

    ASSERT(Nm >= (int)basis.size());

    dst->linear_combination(basis,data);
    
    return PyLong_FromLong(0);
  });
