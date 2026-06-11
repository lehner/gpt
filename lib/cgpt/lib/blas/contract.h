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
class cgpt_contract_job : public cgpt_blas_job_base {
 public:

  /*
    t1[a,b,c]*t2[c,d]*t3[d,e,e]=t0[a,b]

    Dimensions = a,b,c,d,e    (length of each)
    Strides t1 = stride_a, stride_b, stride_c, 0, 0
    Strides t2 = 0, 0, stride_c, stride_d, 0
    Strides t3 = 0, 0, 0, stride_d, stride_e1, stride_e2
    Strides t0 = stride_a, stride_b, 0, 0, 0

    Then parallelize over strides in t0

    Could add flags for CC for each tensor t1, t2, t3
  */
  HostDeviceVector<dtype*> tensors;
  HostDeviceVector<long> strides;
  HostDeviceVector<long> dimensions;
  HostDeviceVector<long> conjugate;
  long ntensors;
  long ndimensions;
  long target_dimensions;
  long total, total_target;
  
  cgpt_contract_job(std::vector<void*>& _tensors,
		    std::vector<std::vector<long>>& _strides,
		    std::vector<long>& _dimensions,
		    std::vector<long>& _conjugate)  {

    // fill tensors
    tensors.resize(_tensors.size());
    for (size_t i=0;i<tensors.size();i++)
      tensors[i] = (dtype*)_tensors[i];
    tensors.toDevice();
    ntensors = (long)tensors.size();
 
    // fill dimensions
    dimensions.resize(_dimensions.size());
    for (size_t i=0;i<dimensions.size();i++)
      dimensions[i] = _dimensions[i];
    dimensions.toDevice();
    ndimensions = (long)dimensions.size();
    
    // fill strides
    ASSERT(_strides.size() == _tensors.size());
    for (auto s : _strides)
      ASSERT(s.size() == _dimensions.size());
    
    strides.resize(ntensors*ndimensions);
    for (long t=0;t<ntensors;t++)
      for (long d=0;d<ndimensions;d++)
	strides[t*ndimensions + d] = _strides[t][d];
    strides.toDevice();

    for (target_dimensions=0;target_dimensions<ndimensions;target_dimensions++)
      if (_strides[0][target_dimensions] == 0)
	break;
    for (long d=target_dimensions;d<ndimensions;d++)
      ASSERT(_strides[0][d] == 0);
    
    // fill conjugate
    conjugate.resize(_conjugate.size());
    ASSERT(_conjugate.size() == _tensors.size());
    for (long t=0;t<ntensors;t++)
      conjugate[t] = _conjugate[t];
    conjugate.toDevice();

    // prepare outer and inner loops
    total = 1;
    total_target = 1;
    for (long d=0;d<ndimensions;d++) {
      total *= dimensions[d];
      if (d < target_dimensions)
	total_target *= dimensions[d];
    }
  }
  
  virtual ~cgpt_contract_job() {
  }

  virtual void execute(GridBLAS& blas) {
    blas.synchronise();

    // auto-capturing "this" does not work, so create local variables
    dtype** _ptensors = tensors.device;
    long* _pdimensions = dimensions.device;
    long* _pstrides = strides.device;
    long* _pconjugate = conjugate.device;
    
    long _ntensors = ntensors;
    long _ndimensions = ndimensions;
     
    long _total_ortho = total / total_target;
    long _target_dimensions = target_dimensions;

    constexpr long max_dimensions = 16;
    ASSERT(_ndimensions <= max_dimensions);

    accelerator_for(target_index,total_target,1, {
	long coor[max_dimensions];

	long idx = target_index;
	long t0_tensor_idx = 0;
	for (long d=_target_dimensions - 1;d>=0;d--) {
	  coor[d] = idx % _pdimensions[d]; idx /= _pdimensions[d];
	  t0_tensor_idx += coor[d] * _pstrides[0*_ndimensions + d];
	}

	dtype acc = 0;
	for (long ortho_index=0;ortho_index<_total_ortho;ortho_index++) {

	  // get remaining coor first
	  long idx = ortho_index;
	  for (long d=_ndimensions-1;d>=_target_dimensions;d--) {
	    coor[d] = idx % _pdimensions[d]; idx /= _pdimensions[d];
	  }

	  // now get the factor
	  dtype fac = 1;
	  for (long t=1;t<_ntensors;t++) {

	    // first my index
	    long tt_tensor_idx = 0;
	    for (long d=0;d<_ndimensions;d++)
	      tt_tensor_idx += coor[d] * _pstrides[t*_ndimensions + d];

	    // then apply
	    auto & r = _ptensors[t][tt_tensor_idx];
	    fac *= (_pconjugate[t]) ? adj(r) : r;
	  }

	  acc += fac;
	}

	_ptensors[0][t0_tensor_idx] = acc;

	// TODO: if multiple contracts had the same dimensions up to target_dimension (and same target_dimension)
	// then I can pull multiple into ortho_index
      });
  }
};
