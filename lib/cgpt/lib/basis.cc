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

EXPORT(qr_decomposition,{

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
    int j0,j1,k0,k1,idx;
    long use_accelerator;
    if (!PyArg_ParseTuple(args, "OOiiiiil", &_basis, &_Qt, &j0, &j1, &k0, &k1, &idx, &use_accelerator)) {
      return NULL;
    }
    
    std::vector<cgpt_Lattice_base*> basis;
    cgpt_basis_fill(basis,_basis,idx);

    ASSERT(basis.size() > 0);

    int Nm, NmP, dtype;
    cgpt_numpy_query_matrix(_Qt, dtype, Nm, NmP);
    ASSERT(Nm == NmP);
    ASSERT(j0 <= j1 && k0 <= k1 && j0 >=0 && k0 >= 0 && k1 <= Nm && j1 <= Nm && 
	   (int)basis.size() >= j1 && (int)basis.size() >= k1);

    if (dtype == NPY_COMPLEX128) {
      ComplexD* data;
      cgpt_numpy_import_matrix(_Qt,data,Nm);
      basis[0]->basis_rotate(basis,data,j0,j1,k0,k1,Nm,use_accelerator);
    } else if (dtype == NPY_FLOAT64) {
      RealD* data;
      cgpt_numpy_import_matrix(_Qt,data,Nm);
      basis[0]->basis_rotate(basis,data,j0,j1,k0,k1,Nm,use_accelerator);
    } else {
      ERR("Unknown dtype");
    }
    
    return PyLong_FromLong(0);
  });

EXPORT(linear_combination,{

    PyObject* _dst, * _basis,* _Qt;
    long basis_n_block;
    if (!PyArg_ParseTuple(args, "OOOl", &_dst, &_basis, &_Qt, &basis_n_block)) {
      return NULL;
    }
    
    std::vector<cgpt_Lattice_base*> basis;
    long basis_n_virtual = cgpt_basis_fill(basis,_basis);

    std::vector<cgpt_Lattice_base*> dst;
    long dst_n_virtual = cgpt_basis_fill(dst,_dst);

    ASSERT(basis.size() > 0 && dst.size() > 0);
    ASSERT(basis_n_virtual == dst_n_virtual);

    ComplexD* data;
    int Nvec,Nm;
    cgpt_numpy_import_matrix(_Qt,data,Nvec,Nm);

    ASSERT((dst.size() / dst_n_virtual) == Nvec);
    ASSERT(Nm*basis_n_virtual == (int)basis.size());

    dst[0]->linear_combination(dst,basis,data,basis_n_virtual,basis_n_block);
    
    return PyLong_FromLong(0);
  });


template<typename T>
bool cgpt_bilinear_combination_helper(std::vector<cgpt_Lattice_base*> & _dst,
				      std::vector<cgpt_Lattice_base*> & _left_basis,
				      std::vector<cgpt_Lattice_base*> & _right_basis,
				      ComplexD* Qt,
				      int32_t* left_indices,
				      int32_t* right_indices,
				      long n_virtual, long Nm) {

  if ( _dst[0]->type() != typeid(T).name() )
    return false;
  
  PVector<Lattice<T>> dst, left_basis, right_basis;
  cgpt_basis_fill(left_basis,_left_basis);
  cgpt_basis_fill(right_basis,_right_basis);
  cgpt_basis_fill(dst,_dst);

  cgpt_bilinear_combination(dst, left_basis, right_basis, Qt, left_indices, right_indices, n_virtual, Nm);

  return true;
}

EXPORT(bilinear_combination,{

    PyObject* _dst, * _left_basis,* _right_basis, * _Qt, * _lidx, * _ridx;
    if (!PyArg_ParseTuple(args, "OOOOOO", &_dst, &_left_basis,&_right_basis, &_Qt, &_lidx, &_ridx)) {
      return NULL;
    }
    
    std::vector<cgpt_Lattice_base*> left_basis, right_basis;
    long basis_n_virtual = cgpt_basis_fill(left_basis,_left_basis);
    ASSERT(basis_n_virtual == cgpt_basis_fill(right_basis,_right_basis));

    std::vector<cgpt_Lattice_base*> dst;
    long dst_n_virtual = cgpt_basis_fill(dst,_dst);

    ASSERT(left_basis.size() > 0 && dst.size() > 0);
    ASSERT(basis_n_virtual == dst_n_virtual);

    ComplexD* data;
    int Nvec,Nm;
    cgpt_numpy_import_matrix(_Qt,data,Nvec,Nm);

    int32_t* lidx;
    int Nvec_lidx,Nm_lidx;
    cgpt_numpy_import_matrix(_lidx,lidx,Nvec_lidx,Nm_lidx);
    ASSERT(Nvec_lidx == Nvec && Nm_lidx == Nm);

    int32_t* ridx;
    int Nvec_ridx,Nm_ridx;
    cgpt_numpy_import_matrix(_ridx,ridx,Nvec_ridx,Nm_ridx);
    ASSERT(Nvec_ridx == Nvec && Nm_ridx == Nm);

    ASSERT((dst.size() / dst_n_virtual) == Nvec);

    if (cgpt_bilinear_combination_helper<iSinglet<vComplexF>>(dst,left_basis,right_basis,data,lidx,ridx,dst_n_virtual,Nm));
    else if (cgpt_bilinear_combination_helper<iSinglet<vComplexD>>(dst,left_basis,right_basis,data,lidx,ridx,dst_n_virtual,Nm));
    else {
      ERR("Type %s unsupported", dst[0]->type().c_str() );
    }

    return PyLong_FromLong(0);
  });

