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

template<class vobj,class CComplex,int nbasis,class VLattice>
  inline void vectorizableBlockProject(Lattice<iVector<CComplex,nbasis >> &coarseData,
				       const Lattice<vobj>   &fineData,
				       const VLattice &Basis)
{
  GridBase * fine  = fineData.Grid();
  GridBase * coarse= coarseData.Grid();

  Lattice<iScalar<CComplex>> ip(coarse);
  Lattice<vobj>     fineDataRed = fineData;

  autoView( coarseData_ , coarseData, AcceleratorWrite);
  autoView( ip_         , ip,         AcceleratorWrite);
  for(int v=0;v<nbasis;v++) {
    blockInnerProductD(ip,Basis[v],fineDataRed); // ip = <basis|fine>
    accelerator_for( sc, coarse->oSites(), vobj::Nsimd(), {
	convertType(coarseData_[sc](v),ip_[sc]);
      });
  }
}

template<class CComplex,class VLattice>
inline void vectorBlockOrthonormalize(Lattice<CComplex> &ip,std::vector<VLattice> &Basis)
{
  GridBase *coarse = ip.Grid();
  GridBase *fine   = Basis[0][0].Grid();

  int       nvec = Basis.size();
  int       nbasis = Basis[0].size() ;

  // checks
  subdivides(coarse,fine);
  for (int j=0;j<nvec;j++){
    for(int i=0;i<nbasis;i++){
      conformable(Basis[j][i].Grid(),fine);
    }
  }

  Lattice<CComplex> sip(coarse);
  typename std::remove_reference<decltype(Basis[0][0])>::type zz(fine);
  zz = Zero();
  zz.Checkerboard()=Basis[0][0].Checkerboard();
  for(int v=0;v<nbasis;v++) {
    for(int u=0;u<v;u++) {
      //Inner product & remove component
      sip = Zero();
      for (int j=0;j<nvec;j++) {
	blockInnerProductD(ip,Basis[j][u],Basis[j][v]);
	sip -= ip;
      }
      for (int j=0;j<nvec;j++)
	blockZAXPY(Basis[j][v],sip,Basis[j][u],Basis[j][v]);
    }

    // block normalize
    sip = Zero();
    for (int j=0;j<nvec;j++) {
      blockInnerProductD(ip,Basis[j][v],Basis[j][v]);
      sip += ip;
    }
    sip = pow(sip,-0.5);
    ip = Zero();
    for (int j=0;j<nvec;j++)
      blockZAXPY(Basis[j][v],sip,Basis[j][v],zz);
  }
}

// NOTE: This is only here for comparison with lut version -- can remove later
template<class vobj,class CComplex,int nbasis,class VLattice>
inline void vectorizableBlockProjectUsingNoLut(Lattice<iVector<CComplex, nbasis>>& coarseData,
                                               const Lattice<vobj>&                fineData,
                                               const VLattice&                     Basis)
{
  GridBase *fine   = fineData.Grid();
  GridBase *coarse = coarseData.Grid();

  int _ndimension = coarse->_ndimension;

  // checks
  assert(nbasis == Basis.size());
  subdivides(coarse, fine);
  for(int i = 0; i < nbasis; i++) {
    conformable(Basis[i], fineData);
  }

  Coordinate block_r(_ndimension);
  Coordinate fine_rdimensions   = fine->_rdimensions;
  Coordinate coarse_rdimensions = coarse->_rdimensions;

  size_t block_v = 1;
  for(int d = 0; d < _ndimension; ++d) {
    block_r[d] = fine->_rdimensions[d] / coarse->_rdimensions[d];
    assert(block_r[d] * coarse->_rdimensions[d] == fine->_rdimensions[d]);
    block_v *= block_r[d];
  }
  assert(block_v == fine->oSites() / coarse->oSites());

  autoView(fineData_v, fineData, AcceleratorRead);
  autoView(coarseData_v, coarseData, AcceleratorWrite);

  typedef decltype(Basis[0].View(AcceleratorRead)) View;
  Vector<View> Basis_v; Basis_v.reserve(Basis.size());
  for(int i=0;i<Basis.size();i++){
    Basis_v.push_back(Basis[i].View(AcceleratorRead));
  }

  accelerator_for(sci, nbasis * coarse->oSites(), vobj::Nsimd(), {
    auto i  = sci % nbasis;
    auto sc = sci / nbasis;

    Coordinate coor_c(_ndimension);
    Lexicographic::CoorFromIndex(coor_c, sc, coarse_rdimensions);

    int sf;
    decltype(innerProductD2(Basis_v[sf](0), fineData_v(sf))) reduce = Zero();

    for(int sb = 0; sb < block_v; ++sb) {
      Coordinate coor_b(_ndimension);
      Coordinate coor_f(_ndimension);

      Lexicographic::CoorFromIndex(coor_b, sb, block_r);
      for(int d = 0; d < _ndimension; ++d) coor_f[d] = coor_c[d] * block_r[d] + coor_b[d];
      Lexicographic::IndexFromCoor(coor_f, sf, fine_rdimensions);

      reduce = reduce + innerProductD2(Basis_v[i](sf), fineData_v(sf));
    }
    convertType(coarseData_v[sc](i), TensorRemove(reduce));
  });

  for(int i=0;i<Basis.size();i++) Basis_v[i].ViewClose();
}
