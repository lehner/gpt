/*
  CGPT

  Authors: Christoph Lehner 2020
  
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
