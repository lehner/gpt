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

  virtual void execute(GridBLAS& blas) = 0;
};

template<typename dtype>
class cgpt_gemm_job : public cgpt_blas_job_base {
 public:
  
  deviceVector<dtype*> BLAS_A, BLAS_B, BLAS_C;
  GridBLASOperation_t opA, opB;
  long m,n,k;
  ComplexD alpha, beta;
  
  void fill_pointers(deviceVector<dtype*>& dst, dtype* base, int64_t* idx, long num, long words) {
    deviceVector<int64_t> d_idx(num);
    acceleratorCopyToDevice(idx, &d_idx[0], num*sizeof(int64_t));
    int64_t* p = &d_idx[0];
    dtype** _dst = &dst[0];
    accelerator_for(idx, num, 1, {
	_dst[idx] = &base[p[idx]*words];
      });
  }

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
		void* _data_A, int64_t* idxA, long _opA,
		void* _data_B, int64_t* idxB, long _opB,
		ComplexD _beta,
		void* _data_C, int64_t* idxC, long num_elements) :
  
    BLAS_A(num_elements),
    BLAS_B(num_elements),
      BLAS_C(num_elements),
      m(_m), n(_n), k(_k), alpha(_alpha), beta(_beta) {
    
    fill_pointers(BLAS_A, (dtype*)_data_A, idxA, num_elements, m*k);
    fill_pointers(BLAS_B, (dtype*)_data_B, idxB, num_elements, k*n);
    fill_pointers(BLAS_C, (dtype*)_data_C, idxC, num_elements, m*n);

    opA = convert_op_code(_opA);
    opB = convert_op_code(_opB);

  }
  
  virtual ~cgpt_gemm_job() {
  }

  virtual void execute(GridBLAS& blas) {
    std::cout << GridLogMessage << "Call batched gemm " << m << ", " << n << ", " << k << std::endl;
    blas.gemmBatched(opA, opB, m, n, k, (dtype)alpha, BLAS_A, BLAS_B, (dtype)beta, BLAS_C);
    //blas.gemmBatched(m, n, k, (dtype)alpha, BLAS_A, BLAS_B, (dtype)beta, BLAS_C);
    std::cout << GridLogMessage << "Done" << std::endl;
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
