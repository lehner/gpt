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

#ifdef GRID_HAS_ACCELERATOR
#define accelerator_foreach_lane( lane, nsimd, ... ) { int lane = acceleratorSIMTlane(nsimd); __VA_ARGS__ }
#else
#define accelerator_foreach_lane( lane, nsimd, ... ) { for (int lane=0;lane<nsimd;lane++){__VA_ARGS__} }
#endif  

template<class vobj> inline void cgpt_pickCheckerboard(int cb,Lattice<vobj> &half,const Lattice<vobj> &full)
{
  half.Checkerboard() = cb;

  autoView( half_v, half, AcceleratorWriteDiscard);
  autoView( full_v, full, AcceleratorRead);

  auto half_p = &half_v[0];
  auto full_p = &full_v[0];

  GridView gf(full.Grid());
  GridView gh(half.Grid());
  
  accelerator_for(ss, full.Grid()->oSites(),full.Grid()->Nsimd(),{
    int cbos;
    Coordinate coor;
    gf.oCoorFromOindex(coor,ss);
    cbos=gh.CheckerBoard(coor);

    if (cbos==cb) {
      int ssh=gh.oIndex(coor);
      coalescedWrite(half_v[ssh], coalescedRead(full_v[ss]));
    }
  });
}

template<class vobj> inline void cgpt_setCheckerboard(Lattice<vobj> &full,const Lattice<vobj> &half)
{
  int cb = half.Checkerboard();
  autoView( half_v , half, AcceleratorRead);
  autoView( full_v , full, AcceleratorWrite);

  auto half_p = &half_v[0];
  auto full_p = &full_v[0];

  GridView gf(full.Grid());
  GridView gh(half.Grid());

  accelerator_for(ss,full.Grid()->oSites(),full.Grid()->Nsimd(),{

    Coordinate coor;
    int cbos;

    gf.oCoorFromOindex(coor,ss);
    cbos=gh.CheckerBoard(coor);
      
    if (cbos==cb) {
      int ssh=gh.oIndex(coor);
      coalescedWrite(full_v[ss],coalescedRead(half_v[ssh]));
    }
  });
}

template<class VobjOut, class VobjIn> void cgpt_precisionChange(Lattice<VobjOut> &out, const Lattice<VobjIn> &in)
{
  ASSERT(out.Grid()->Nd() == in.Grid()->Nd());
  for(int d=0;d<out.Grid()->Nd();d++){
    ASSERT(out.Grid()->FullDimensions()[d] == in.Grid()->FullDimensions()[d]);
  }
  out.Checkerboard() = in.Checkerboard();
  GridBase *in_grid=in.Grid();
  GridBase *out_grid = out.Grid();

  typedef typename VobjOut::scalar_type out_t;
  typedef typename VobjIn::scalar_type in_t;
  constexpr int n_elem = GridTypeMapper<VobjOut>::count;
  ASSERT(n_elem == GridTypeMapper<VobjIn>::count);
    
  autoView( out_v , out, AcceleratorWriteDiscard);
  autoView( in_v , in, AcceleratorRead);
  auto out_p = &out_v[0];
  auto in_p = &in_v[0];

  GridView go(out_grid);
  GridView gi(in_grid);

  int ndim = out_grid->Nd();

  Vector<Coordinate> _in_icoor(in_grid->Nsimd());
  Coordinate* in_icoor = &_in_icoor[0];

  for(int lane=0; lane < in_grid->Nsimd(); lane++){
    in_icoor[lane].resize(ndim);
    in_grid->iCoorFromIindex(in_icoor[lane], lane);
  }

  int in_nsimd = in_grid->Nsimd();
  int out_nsimd = out_grid->Nsimd();

  accelerator_for(in_oidx,in_grid->oSites(),in_nsimd,{

      Coordinate in_ocoor(ndim);
      int lcoor;
      
      gi.oCoorFromOindex(in_ocoor, in_oidx);

      accelerator_foreach_lane(in_lane,in_nsimd,{

	  uint64_t out_lane = 0;
	  uint64_t out_oidx = 0;
	  
	  for(int mu=0;mu<ndim;mu++) {
	    lcoor = in_ocoor[mu] + gi._rdimensions[mu]*in_icoor[in_lane][mu];
	    out_lane += go._istride[mu] * (lcoor / go._rdimensions[mu]);
	    out_oidx += go._ostride[mu] * (lcoor % go._rdimensions[mu]);
	  }

	  for (int i=0;i<n_elem;i++) {
	    ((out_t*)&out_p[out_oidx])[i * out_nsimd + out_lane] = ((in_t*)&in_p[in_oidx])[i * in_nsimd + in_lane];
	  }
	  
	});
	
  });
}
