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

class GridView {
 public:
  GridView() = default;

  Coordinate _rdimensions;
  Coordinate _ldimensions;
  Coordinate _ostride;
  Coordinate _istride;
  Coordinate _checker_dim_mask;
  int _checker_dim;
  
  GridView(GridBase* grid) {
    _rdimensions = grid->_rdimensions;
    _ldimensions = grid->_ldimensions;
    _checker_dim_mask = grid->_checker_dim_mask;
    _ostride = grid->_ostride;
    _istride = grid->_istride;
    if (grid->_isCheckerBoarded)
      _checker_dim = ((GridRedBlackCartesian*)grid)->_checker_dim;
    else
      _checker_dim = -1;
  }
  
  
  accelerator_inline void oCoorFromOindex (Coordinate& coor,int Oindex) {
    Lexicographic::CoorFromIndex(coor,Oindex,_rdimensions);
  }

  accelerator_inline int oIndex(Coordinate &coor) {
    int idx = 0;
    for (int d = 0; d < coor.size(); d++) {
      if (d == _checker_dim) {
	idx += _ostride[d] * ((coor[d] / 2) % _rdimensions[d]);
      } else {
	idx += _ostride[d] * (coor[d] % _rdimensions[d]);
      }
    }
    return idx;
  }

  accelerator_inline int iIndex(Coordinate &lcoor) {
    int idx = 0;
    for (int d = 0; d < lcoor.size(); d++) {
      if (d == _checker_dim) {
	idx += _istride[d] * (lcoor[d] / (2 * _rdimensions[d]));
      } else {
	idx += _istride[d] * (lcoor[d] / _rdimensions[d]);
      }
    }
    return idx;
  }


  accelerator_inline int CheckerBoard(const Coordinate &site) {
    int linear=0;
    for(int d=0;d<site.size();d++){ 
      if(_checker_dim_mask[d])
	linear=linear+site[d];
    }
    return (linear&0x1);
  }

};
