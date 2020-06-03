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
static PyObject* cgpt_memory_view_coordinates(GridBase* grid, int cb) {
  int Nd = grid->Nd();
  int rank = grid->_processor;
  std::vector<long> dim(2);
  dim[0] = (long)grid->iSites() * (long)grid->oSites();
  dim[1] = Nd;
  int cb_dim = -1;
  for (int i=0;i<Nd;i++)
    if (grid->CheckerBoarded(i)) { // always use first checkerboarded direction as cb direction
      cb_dim=i;
      break;
    }
  
  PyArrayObject* a = (PyArrayObject*)PyArray_SimpleNew((int)dim.size(), &dim[0], NPY_INT32);
  int32_t* s = (int32_t*)PyArray_DATA(a);
  
  if (cb_dim == -1) {
    thread_for(osite,grid->oSites(),{
	Coordinate gcoor(Nd);
	for (long isite=0;isite<grid->iSites();isite++) {
	  long idx = osite * grid->iSites() + isite;
	  grid->RankIndexToGlobalCoor(rank,osite,isite,gcoor);
	  for (int i=0;i<Nd;i++)
	    s[Nd*idx + i] = gcoor[i];
	}
      });
  } else {
    thread_for(osite,grid->oSites(),{
	Coordinate gcoor(Nd);
	for (long isite=0;isite<grid->iSites();isite++) {
	  long idx = osite * grid->iSites() + isite;
	  grid->RankIndexToGlobalCoor(rank,osite,isite,gcoor);
	  gcoor[cb_dim] *= 2;
	  if ( cb != grid->CheckerBoard(gcoor) )
	    gcoor[cb_dim] += 1;
	  for (int i=0;i<Nd;i++)
	    s[Nd*idx + i] = gcoor[i];
	}
      });
  }
  
  PyArray_CLEARFLAGS(a,NPY_ARRAY_WRITEABLE); // read-only, so we can cache distribute plans
  return (PyObject*)a;
}
