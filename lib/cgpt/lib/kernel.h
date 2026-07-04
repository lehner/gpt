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
class cgpt_kernel_job_base {
 public:
  virtual ~cgpt_kernel_job_base() {
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

  virtual std::string description() = 0;
};

#include "kernel/gemm.h"
#include "kernel/det.h"
#include "kernel/inv.h"
#include "kernel/fft.h"
#include "kernel/copy.h"
#include "kernel/accumulate.h"
#include "kernel/indexed_sum.h"
#include "kernel/contract.h"
#include "kernel/transpose.h"


class cgpt_kernel {
 public:

  std::vector<cgpt_kernel_job_base*> jobs;
  std::vector<std::string> desc;
  GridBLAS blas;

  cgpt_kernel() {
  }

  ~cgpt_kernel() {
    for (auto j : jobs)
      delete j;
  }

  void ensure_desc() {
    if (desc.size() != jobs.size()) {
      desc.resize(jobs.size());
      for (size_t i=0;i<jobs.size();i++)
	desc[i] = jobs[i]->description();
    }
  }
  
  void execute() {
    ensure_desc();
    
    for (size_t i=0;i<jobs.size();i++) {
      Timer(desc[i]);
      jobs[i]->execute(blas);
    }

    Timer();

    blas.synchronise();
  }

  std::string str() {
    ensure_desc();
	
    std::ostringstream oss;
    for (size_t i=0;i<jobs.size();i++)
      oss << "Step[" << i << "] = " << desc[i] << std::endl;
    return oss.str();
  }
};
