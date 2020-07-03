/*
    GPT - Grid Python Toolkit
    Copyright (C) 2020  Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)

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
template<typename coor_t>
Coordinate toCanonical(const coor_t svec) {
  if (svec.size() == 4) {
    Coordinate cvec(4);
    cvec[0] = svec[0];
    cvec[1] = svec[1];
    cvec[2] = svec[2];
    cvec[3] = svec[3];
    return cvec;
  } else if (svec.size() == 5) {
    Coordinate cvec(5);
    cvec[0] = svec[1];
    cvec[1] = svec[2];
    cvec[2] = svec[3];
    cvec[3] = svec[4];
    cvec[4] = svec[0];
    return cvec;
  } else {
    ERR("Dimension %d not supported",(int)svec.size());
  }
}

template<typename coor_t>
Coordinate fromCanonical(const coor_t cvec) {
  if (cvec.size() == 4) {
    Coordinate svec(4);
    svec[0] = cvec[0];
    svec[1] = cvec[1];
    svec[2] = cvec[2];
    svec[3] = cvec[3];
    return svec;
  } else if (cvec.size() == 5) {
    Coordinate svec(5);
    svec[0] = cvec[4];
    svec[1] = cvec[0];
    svec[2] = cvec[1];
    svec[3] = cvec[2];
    svec[4] = cvec[3];
    return svec;
  } else {
    ERR("Dimension %d not supported",(int)cvec.size());
  }
}

struct cgpt_order_lexicographic {
  template<typename coor_t>
  inline void operator()(coor_t& coor, long idx, const coor_t& size) {
    Lexicographic::CoorFromIndex(coor,idx,size);
  }
};

struct cgpt_order_reverse_lexicographic {
  template<typename coor_t>
  inline void operator()(coor_t& coor, long idx, const coor_t& size) {
    Lexicographic::CoorFromIndexReversed(coor,idx,size);
  }
};

struct cgpt_order_canonical {
  template<typename coor_t>
  inline void operator()(coor_t& coor, long idx, const coor_t& c_size) {
    coor_t c_coor(coor.size());
    Lexicographic::CoorFromIndex(c_coor,idx,c_size);
    coor = fromCanonical(c_coor);
  }
};

template<typename order_t, typename coor_t>
  inline bool cgpt_fill_cartesian_view_coordinates(int32_t* d,int Nd,
						   const std::vector<long>& top,
						   const coor_t& size,
						   const std::vector<long>& checker_dim_mask,
						   long fstride,int cbf,int cb,
						   long points, order_t order) {
  bool first_on_grid = false;
  thread_region
    {
      coor_t coor(Nd);
      thread_for_in_region(idx,points,{
	  order(coor,idx,size);
	  long idx_cb = (idx % fstride) + ((idx / fstride)/cbf) * fstride;
	  long site_cb = 0;
	  for (int i=0;i<Nd;i++)
	    if (checker_dim_mask[i])
	      site_cb += top[i] + coor[i];
	  if (site_cb % 2 == cb) {
	    for (int i=0;i<Nd;i++)
	      d[Nd*idx_cb + i] = top[i] + coor[i];
	    if (!idx)
	      first_on_grid = true;
	  }
	});
    }
  return first_on_grid;
}
