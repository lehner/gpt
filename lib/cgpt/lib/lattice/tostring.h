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

  
    This extends Grid's Lattice::operator<< to deal with checkerboards.
*/
template<typename T>
std::string cgpt_lattice_to_str(Lattice<T>& l) {

  typedef typename Lattice<T>::vector_object vobj;
  typedef typename vobj::scalar_object sobj;

  std::stringstream stream;

  Coordinate gcoor;
  GridBase* grid = l.Grid();
  for(int site=0;site<grid->_fsites;site++){

    Lexicographic::CoorFromIndex(gcoor,site,grid->_fdimensions);

    if (l.Checkerboard() == grid->CheckerBoard(gcoor)) {

      sobj ss;
      peekSite(ss,l,gcoor);
      stream<<"[";
      for(int d=0;d<gcoor.size();d++){
	stream<<gcoor[d];
	if(d!=gcoor.size()-1) stream<<",";
      }
      stream<<"]\t";
      stream<<ss<<std::endl;
    }
  }

  return stream.str();
}
