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

  virtual std::string description() = 0;
};

#include "blas/gemm.h"
#include "blas/det.h"
#include "blas/inv.h"
#include "blas/accumulate.h"
#include "blas/indexed_sum.h"
#include "blas/contract.h"
#include "blas/transpose.h"


class cgpt_blas {
 public:

  std::vector<cgpt_blas_job_base*> jobs;
  std::vector<std::string> desc;
  GridBLAS blas;

  cgpt_blas() {
  }

  ~cgpt_blas() {
    for (auto j : jobs)
      delete j;
  }

  void execute() {
    if (!desc.size()) {
      desc.resize(jobs.size());
      for (size_t i=0;i<jobs.size();i++)
	desc[i] = jobs[i]->description();
    }
    
    for (size_t i=0;i<jobs.size();i++) {
      Timer(desc[i]);
      jobs[i]->execute(blas);
    }

    Timer();

    blas.synchronise();
  }
};
