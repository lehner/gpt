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

  std::string description() {
    std::ostringstream oss;
    oss << "Accumulate(" << n << ") x " << BLAS_A.size();
    return oss.str();
  }

  virtual void execute(GridBLAS& blas) {
    constexpr int Nsimd = sizeof(vComplexF) / sizeof(ComplexF);
    dtype** p = &BLAS_A[0];
    ASSERT(n % Nsimd == 0);
    long m = BLAS_A.size();

    blas.synchronise();

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
