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


    This code is based on original Grid code.
*/
static inline void cgpt_CoorFromOuterIndex(Coordinate& Coor, uint64_t site, GridBase* grid, int parity) {
  int cd = grid->_checker_dim;
  int ortho_parity = -1;
  Lexicographic::CoorFromIndex(Coor,site,grid->_rdimensions);
  if (cd != -1) {
    ortho_parity = 0;
    for (int j=0;j<Coor.size();j++)
      if (j != cd)
	ortho_parity += Coor[j];
      
    // x_real = x * 2 + (ortho_parity + parity) % 2   ->  x' = x + shift
    Coor[cd] = Coor[cd] * 2 + (ortho_parity + parity) % 2;
  }
}

static inline uint64_t cgpt_OuterIndexFromCoor(Coordinate& Coor, GridBase* grid, int parity) {
  int cd = grid->_checker_dim;
  int ortho_parity = -1;
  if (cd != -1) {
    ortho_parity = 0;
    for (int j=0;j<Coor.size();j++)
      if (j != cd)
	ortho_parity += Coor[j];
  }

  uint64_t offset = 0;
  for (int j=0;j<Coor.size();j++) {
    int cc = Coor[j];
    if (j == cd) {
      cc = cc - (ortho_parity + parity) % 2;
      cc /= 2;
    }
    // x_real = x * 2 + (ortho_parity + parity) % 2   ->  x' = x + shift
    offset += grid->_ostride[j]*cc;
  }
  return offset;
}

class cgpt_GeneralLocalStencil : public GeneralLocalStencilView {
public:
  typedef GeneralLocalStencilView View_type;

protected:
  GridBase *                        _grid;

public: 
  GridBase *Grid(void) const { return _grid; }

  View_type View(int mode) const {
    View_type accessor(*( (View_type *) this));
    return accessor;
  }

  // Resident in managed memory
  Vector<GeneralStencilEntry>  _entries;

  cgpt_GeneralLocalStencil(GridBase *grid, const std::vector<Coordinate> &shifts, int parity)
  {
    int npoints = shifts.size();
    int osites  = grid->oSites();

    if (grid->_isCheckerBoarded)
      ASSERT(parity != -1);
    
    this->_grid    = grid;
    this->_npoints = npoints;
    this->_entries.resize(npoints* osites);
    this->_entries_p = &_entries[0];

    thread_for(site, osites, {
	Coordinate Coor;
	Coordinate NbrCoor(grid->Nd());
	int cd = grid->_checker_dim;

	for(Integer ii=0;ii<npoints;ii++){
	  Integer lex = site*npoints+ii;
	  GeneralStencilEntry SE;

	  ////////////////////////////////////////////////
	  // Outer index of neighbour Offset calculation
	  ////////////////////////////////////////////////
	  cgpt_CoorFromOuterIndex(Coor,site,grid,parity);
	  
	  for(int d=0;d<Coor.size();d++){
	    int rd = grid->_rdimensions[d];
	    if (cd == d)
	      rd *= 2;

	    ASSERT(shifts[ii][d] >= -rd);
	    NbrCoor[d] = (Coor[d] + shifts[ii][d] + rd )%rd;
	  }

	  SE._offset = cgpt_OuterIndexFromCoor(NbrCoor,grid,parity);
	  
	  ////////////////////////////////////////////////
	  // Inner index permute calculation
	  // Simpler version using icoor calculation
	  ////////////////////////////////////////////////
	  SE._permute =0;
	  SE._wrap=0;
	  for(int d=0;d<Coor.size();d++){

	    int fd = grid->_fdimensions[d];
	    int rd = grid->_rdimensions[d];
	    int ld = grid->_ldimensions[d];
	    int ly = grid->_simd_layout[d];

	    if (cd == d)
	      rd *= 2;

	    assert((ly==1)||(ly==2)||(ly==grid->Nsimd()));

	    int shift = (shifts[ii][d]+fd)%fd;  // make it strictly positive 0.. L-1
	    int x = Coor[d];                // x in [0... rd-1] as an oSite 

	    if ( (x + shift)%fd != (x+shift)%ld ){
	      SE._wrap = 1;
	    }
	    
	    int permute_dim  = grid->PermuteDim(d);
	    int permute_slice=0;
	    if(permute_dim){    
	      int  num = shift%rd; // Slice within dest osite cell of slice zero
	      int wrap = shift/rd; // Number of osite local volume cells crossed through
	      // x+num < rd dictates whether we are in same permute state as slice 0
	      if ( x< rd-num ) permute_slice=wrap;
	      else             permute_slice=(wrap+1)%ly;
	    }
	    if ( permute_slice ) {
	      int ptype       =grid->PermuteType(d);
	      uint8_t mask    =0x1<<ptype;
	      SE._permute    |= mask;
	    }
	  }	
	  ////////////////////////////////////////////////
	  // Store in look up table
	  ////////////////////////////////////////////////
	  this->_entries[lex] = SE;

	}
    });
  }
  
};
