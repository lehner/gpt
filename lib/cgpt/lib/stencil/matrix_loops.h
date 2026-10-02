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


    Kernel loops of cgpt_stencil_matrix<T>::execute without communication
    in the kernel (local and general stencils); included once per stencil
    kind with fetch_shifted(obj, factor, site) defined.
*/
      if (has_temporaries) {

	for (uint64_t s0 = 0; s0 < osites; s0 += block) {
	  uint64_t nsites = std::min(block, osites - s0);
	  accelerator_for(ss_in_block,nsites,T::Nsimd(),{

	      uint64_t ss = s0 + ss_in_block;

	      for (int i=0;i<n_code;i++) {

		obj_t t;

		const auto _f0 = &p_code[i].factor[0];
		fetch_local(t, _f0, ss, ss_in_block);

		for (int j=1;j<p_code[i].size;j++) {
		  obj_t f;
		  const auto _f = &p_code[i].factor[j];
		  fetch_local(f, _f, ss, ss_in_block);
		  t = t * f;
		}

		if (!p_code[i].unit_weight)
		  t = p_code[i].weight * t;
		if (p_code[i].accumulate != -1)
		  t += coalescedRead(p_code[i].accumulate_temporary ?
				     p_temp[p_code[i].accumulate * block + ss_in_block] :
				     fields_v[p_code[i].accumulate][ss]);
		coalescedWrite(p_code[i].target_temporary ?
			       p_temp[p_code[i].target * block + ss_in_block] :
			       fields_v[p_code[i].target][ss], t);
	      }

	    });
	}

      } else {

      accelerator_for(ss_block,osites * _npb,T::Nsimd(),{
	  
          uint64_t ss, oblock;
					      
	  MAP_INDEXING(ss, oblock);
	  
	  for (int iblock=0;iblock<_npbs;iblock++) {
	    
	    int i = oblock * _npbs + iblock;
	    
	    obj_t t;
	    
	    const auto _f0 = &p_code[i].factor[0];
	    fetch_local(t, _f0, ss, ss);
	    
	    for (int j=1;j<p_code[i].size;j++) {
	      obj_t f;
	      const auto _f = &p_code[i].factor[j];
	      fetch_local(f, _f, ss, ss);
	      t = t * f;
	    }
	    
	    if (!p_code[i].unit_weight)
	      t = p_code[i].weight * t;
	    if (p_code[i].accumulate != -1)
	      t += coalescedRead(fields_v[p_code[i].accumulate][ss]);
	    coalescedWrite(fields_v[p_code[i].target][ss], t);
	  }
	  
	});

      }
