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
struct _grid_cache_entry_ {
  GridBase* grid;
  int srank, sranks, ref;
};

extern std::map<std::string,_grid_cache_entry_> cgpt_grid_cache;
extern std::map<GridBase*,std::string> cgpt_grid_cache_tag;

static void cgpt_grid_get_info(GridBase* grid, long& srank, long& sranks) {
  auto & c = cgpt_grid_cache[cgpt_grid_cache_tag[grid]];
  srank = c.srank;
  sranks = c.sranks;
}

static std::string cgpt_grid_tag(const Coordinate& fdimensions, 
				 const Coordinate& simd, 
				 const Coordinate& cb_mask, 
				 const Coordinate& mpi, 
				 GridBase* parent) {

  std::string tag = cgpt_str(fdimensions) + "," + cgpt_str(simd) + "," + cgpt_str(cb_mask) + "," + cgpt_str(mpi);
  if (parent) {
    std::string parent_tag = cgpt_grid_cache_tag[parent];
    ASSERT(parent_tag.size() > 0);
    tag = tag + ":" + parent_tag;
  }
  //std::cout << GridLogMessage << "Tag:" << tag << std::endl;
  return tag;

}

static GridBase* cgpt_create_grid(const Coordinate& fdimensions, 
				  const Coordinate& simd,
				  const Coordinate& cb_mask, 
				  const Coordinate& mpi, 
				  GridBase* parent) {
  
  GridBase* grid = 0;
  GridCartesian* grid_full = 0;
  int srank, sranks;
  std::string tag = cgpt_grid_tag(fdimensions,simd,cb_mask,mpi,parent);
  
  auto rg = cgpt_grid_cache.find(tag);
  if (rg != cgpt_grid_cache.end()) {
    rg->second.ref++;
    //std::cout << GridLogMessage << "Reuse" << rg->second.grid << std::endl;
    return rg->second.grid;
  }

  // is cb ?
  int checker_dim = -1;
  for (int i=0;i<cb_mask.size();i++)
    if (cb_mask[i]) {
      checker_dim = i;
      break;
    }

  // is split ?
  srank=0;
  sranks=1;
  if (parent) {
    ASSERT(parent->_processors.size() == mpi.size());
    for (long i=0;i<mpi.size();i++) {
      ASSERT( parent->_processors[i] % mpi[i] == 0 );
      sranks *= parent->_processors[i] / mpi[i];
    }

    grid_full = new GridCartesian(fdimensions,simd,mpi,*(GridCartesian*)parent,srank);
  } else {
    grid_full = new GridCartesian(fdimensions,simd,mpi);
  }

  if (checker_dim != -1) {
    grid = new GridRedBlackCartesian(grid_full,cb_mask,checker_dim);
    delete grid_full;
  } else {
    grid = grid_full;
  }

  auto & c = cgpt_grid_cache[tag];
  c.ref = 1;
  c.grid = grid;
  c.srank = srank;
  c.sranks = sranks;

  //std::cout << GridLogMessage << "New Grid" << grid << std::endl;

  cgpt_grid_cache_tag[grid] = tag;

  return grid;
}

static void cgpt_delete_grid(GridBase* grid) {
  auto tag = cgpt_grid_cache_tag.find(grid);
  auto c = cgpt_grid_cache.find(tag->second);
  c->second.ref--;
  if (c->second.ref == 0) {
    //std::cout << GridLogMessage << "Delete" << grid << std::endl;
    delete grid;
    cgpt_grid_cache_tag.erase(tag);
    cgpt_grid_cache.erase(c);
  }
}
