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
class cgpt_blas_job_base {
 public:
  virtual ~cgpt_blas_job_base() {
  }

  template<typename dtype>
  void fill_pointers(dtype** _dst, dtype* base, int64_t* idx, long num, long words) {
    deviceVector<int64_t> d_idx(num);
    acceleratorCopyToDevice(idx, &d_idx[0], num*sizeof(int64_t));
    int64_t* p = &d_idx[0];
    accelerator_for(idx, num, 1, {
	_dst[idx] = &base[p[idx]*words];
      });
  }

  virtual void execute(GridBLAS& blas) = 0;
};

template<typename dtype>
class cgpt_gemm_job : public cgpt_blas_job_base {
 public:
  
  deviceVector<dtype*> BLAS_A, BLAS_B, BLAS_C;
  GridBLASOperation_t opA, opB;
  long m,n,k;
  ComplexD alpha, beta;
  GridBLASPrecision_t precision;  
  
  GridBLASOperation_t convert_op_code(long op) {
    switch (op) {
    case 0:
      return GridBLAS_OP_N;
    case 1: // T
      return GridBLAS_OP_T;
    case 3: // T|C == H
      return GridBLAS_OP_C;
    default:
      ERR("Invalid op code %d\n", (int)op);
    }
  }

  cgpt_gemm_job(long _m, long _n,long _k,
		ComplexD _alpha,
		std::vector<void*>& _data_A, std::vector<int64_t*>& idxA, long _opA,
		std::vector<void*>& _data_B, std::vector<int64_t*>& idxB, long _opB,
		ComplexD _beta,
		std::vector<void*>& _data_C, std::vector<int64_t*>& idxC, std::vector<long> num_elements,
		std::string _precision) :
    m(_m), n(_n), k(_k), alpha(_alpha), beta(_beta) {

    // total number of elements
    long total_num_elements = 0;
    for (long elements : num_elements)
      total_num_elements += elements;
    
    BLAS_A.resize(total_num_elements);
    BLAS_B.resize(total_num_elements);
    BLAS_C.resize(total_num_elements);
    
    total_num_elements = 0;
    for (int i=0;i<(int)num_elements.size();i++) {

      fill_pointers(&BLAS_A[total_num_elements], (dtype*)_data_A[i], idxA[i], num_elements[i], m*k);
      fill_pointers(&BLAS_B[total_num_elements], (dtype*)_data_B[i], idxB[i], num_elements[i], k*n);
      fill_pointers(&BLAS_C[total_num_elements], (dtype*)_data_C[i], idxC[i], num_elements[i], m*n);

      total_num_elements += num_elements[i];
    }

    opA = convert_op_code(_opA);
    opB = convert_op_code(_opB);

    if (_precision == "default") {
      precision = GridBLAS_PRECISION_DEFAULT;
    } else if (_precision == "16F") {
      precision = GridBLAS_PRECISION_16F;
    } else if (_precision == "16BF") {
      precision = GridBLAS_PRECISION_16BF;
    } else if (_precision == "TF32") {
      precision = GridBLAS_PRECISION_TF32;
    } else {
      ERR("Unknown precision for compute of gemm: %s", _precision.c_str());
    }
  }
  
  virtual ~cgpt_gemm_job() {
  }

  virtual void execute(GridBLAS& blas) {
    blas.gemmBatched(opA, opB, m, n, k, (dtype)alpha, BLAS_A, BLAS_B, (dtype)beta, BLAS_C, precision);
  }
};

template<typename dtype>
class cgpt_det_job : public cgpt_blas_job_base {
 public:
  
  deviceVector<dtype*> BLAS_A, BLAS_C;
  long n;
  
  cgpt_det_job(long _n,
	       void* _data_A, int64_t* idxA,
	       void* _data_C, int64_t* idxC,
	       long num_elements) :
    
    BLAS_A(num_elements),
    BLAS_C(num_elements),
    n(_n) {
    
    fill_pointers(&BLAS_A[0], (dtype*)_data_A, idxA, num_elements, n*n);
    fill_pointers(&BLAS_C[0], (dtype*)_data_C, idxC, num_elements, 1);

  }
  
  virtual ~cgpt_det_job() {
  }

  virtual void execute(GridBLAS& blas) {
    blas.determinantBatched(n, BLAS_A, BLAS_C);
  }
};

template<typename dtype>
class cgpt_inv_job : public cgpt_blas_job_base {
 public:
  
  deviceVector<dtype*> BLAS_A, BLAS_C;
  long n;
  
  cgpt_inv_job(long _n,
		void* _data_A, int64_t* idxA,
		void* _data_C, int64_t* idxC,
		long num_elements) :
  
    BLAS_A(num_elements),
    BLAS_C(num_elements),
    n(_n) {
    
    fill_pointers(&BLAS_A[0], (dtype*)_data_A, idxA, num_elements, n*n);
    fill_pointers(&BLAS_C[0], (dtype*)_data_C, idxC, num_elements, n*n);

  }
  
  virtual ~cgpt_inv_job() {
  }

  virtual void execute(GridBLAS& blas) {
    blas.inverseBatched(n, BLAS_A, BLAS_C);
  }
};

template<typename dtype>
class cgpt_accumulate_job : public cgpt_blas_job_base {
 public:
  
  deviceVector<dtype*> BLAS_A;
  long n;
  
  cgpt_accumulate_job(long _n,
		      std::vector<void*>& _data_A) :
    BLAS_A(_data_A.size()),
    n(_n) {

    acceleratorCopyToDevice(&_data_A[0], &BLAS_A[0], sizeof(dtype*)*BLAS_A.size());

  }
  
  virtual ~cgpt_accumulate_job() {
  }

  virtual void execute(GridBLAS& blas) {
    constexpr int Nsimd = sizeof(vComplexF) / sizeof(ComplexF);
    dtype** p = &BLAS_A[0];
    ASSERT(n % Nsimd == 0);
    long m = BLAS_A.size();

    accelerator_for(i,n/Nsimd,Nsimd,{
#ifdef GRID_SIMT
	long j = acceleratorSIMTlane(Nsimd);
#else
	for (long j=0;j<Nsimd;j++) {
#endif
	long l = i * Nsimd + j;
	for (long k=1;k<m;k++)
	  p[0][l] += p[k][l];
#ifndef GRID_SIMT
	}
#endif
      });
  }
};

class cgpt_blas {
 public:

  std::vector<cgpt_blas_job_base*> jobs;
  GridBLAS blas;

  cgpt_blas() {
  }

  ~cgpt_blas() {
    for (auto j : jobs)
      delete j;
  }

  void execute() {
    for (auto j : jobs)
      j->execute(blas);

    blas.synchronise();
  }
};
