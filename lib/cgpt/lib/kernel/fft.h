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
class cgpt_fft_job : public cgpt_kernel_job_base {
 public:
  
  long count, size, sign;

  typedef typename FFTW<dtype>::FFTW_plan FFTW_plan;
  typedef typename FFTW<dtype>::FFTW_scalar scalar;

  scalar* s, *d;
  FFTW_plan plan;
  
  cgpt_fft_job(void* _s, void* _d, long _count, long _size, long _sign) :
    s((scalar*)_s), d((scalar*)_d), count(_count), size(_size), sign(_sign) {

    ASSERT(size < ((long)1 << 31));
    int n[]     = {(int)size};

    plan = FFTW<dtype>::fftw_plan_many_dft(1, n, count, s, n, 1, size, d, n, 1, size, (sign < 0) ? FFTW_FORWARD : FFTW_BACKWARD,  FFTW_ESTIMATE);
  }

  std::string description() {
    std::ostringstream oss;
    oss << "FFT(count=" << count << ", size=" << size << ", sign=" << sign << ")";
    return oss.str();
  }
  
  virtual ~cgpt_fft_job() {
    FFTW<dtype>::fftw_destroy_plan(plan);
  }

  virtual void execute(GridBLAS& blas) {
    blas.synchronise();
    FFTW<dtype>::fftw_execute_dft(plan,s,d,(sign < 0) ? FFTW_FORWARD : FFTW_BACKWARD);
  }
};
