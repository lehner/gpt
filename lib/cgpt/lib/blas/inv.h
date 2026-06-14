/*
    GPT - Grid Python Toolkit
    Copyright (C) 2026  Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)

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

  std::string description() {
    std::ostringstream oss;
    oss << "Inv(" << n << ") x " << BLAS_A.size();
    return oss.str();
  }

  virtual void execute(GridBLAS& blas) {
    blas.inverseBatched(n, BLAS_A, BLAS_C);
  }
};
