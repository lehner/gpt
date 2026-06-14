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
class cgpt_transpose_device_memory_view_job : public cgpt_blas_job_base {
 public:

  void* dst, *src;
  std::vector<long> shape, axes;
  
  cgpt_transpose_device_memory_view_job(void* _dst, void* _src, std::vector<long>& _shape, std::vector<long>& _axes) :
    dst(_dst), src(_src), shape(_shape), axes(_axes) { }
  
  virtual ~cgpt_transpose_device_memory_view_job() { }

  virtual void execute(GridBLAS& blas) {
    blas.synchronise();
    cgpt_transpose_device_memory_view<dtype>(dst, src, shape, axes);
  }
};
