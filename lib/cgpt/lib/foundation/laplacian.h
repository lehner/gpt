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

// Laplace operator:
//   laplace(U,D) \psi(x) = \sum_{\mu \in D} (U_\mu(x) \psi(x+\mu) + U_\mu^\dag(x-\mu) \psi(x-\mu) - 2 \psi(x))

template<class Impl>
class Laplacian : public FermionOperator<Impl>
{
public:
  INHERIT_IMPL_TYPES(Impl);

  Laplacian(GaugeField& _Umu,
            GridCartesian& Fgrid,
            GridRedBlackCartesian& Hgrid,
            std::vector<int> dims,
            const ImplParams &p = ImplParams()
            )
    : FermionOperator<Impl>(p) // for parsing boundary phases (and twists)
    , _grid(&Fgrid)
    , _cbgrid(&Hgrid)
    , _tmp(&Hgrid)
    , dimensions(dims)
    , dims_gauge(set_dims_gauge(dims))
    , directions(set_directions(dims))
    , displacements(set_displacements(dims))
    , npoint(2 * dims.size())
    , Umu(&Fgrid)
    , UmuEven(&Hgrid)
    , UmuOdd(&Hgrid)
    , Stencil(&Fgrid, npoint, Even, directions, displacements, 0)
    , StencilEven(&Hgrid, npoint, Even, directions, displacements, 0)
    , StencilOdd(&Hgrid, npoint, Odd, directions, displacements, 0)
  {
    ImportGauge(_Umu);
  }

  std::vector<int> set_directions(std::vector<int> dims)
  {
    std::vector<int> ret(2 * dims.size());
    for (int ii = 0; ii < dims.size(); ii++)
    {
      ret[ii] = dims[ii];
      ret[ii + dims.size()] = dims[ii];
    }
    return ret;
  }

  std::vector<int> set_displacements(std::vector<int> dims)
  {
    std::vector<int> ret(2 * dims.size());
    for (int ii = 0; ii < dims.size(); ii++)
    {
      ret[ii] = 1;
      ret[ii + dims.size()] = -1;
    }
    return ret;
  }

  std::vector<int> set_dims_gauge(std::vector<int> dims)
  {
    std::vector<int> ret(2 * dims.size());
    for (int ii = 0; ii < dims.size(); ii++)
    {
      ret[ii] = dims[ii];
      ret[ii + dims.size()] = dims[ii] + 4;
    }
    return ret;
  }

  // implement the abstract base
  GridBase*     GaugeGrid()           { return _grid; }
  GridBase*     GaugeRedBlackGrid()   { return _cbgrid; }
  GridBase*     FermionGrid()         { return _grid; }
  GridBase*     FermionRedBlackGrid() { return _cbgrid; }
  FermionField& tmp()                 { return _tmp; }

  void ImportGauge(const GaugeField& _Umu) {
  // copied from WilsonFermion
    GaugeField HUmu(_Umu.Grid());
    HUmu = _Umu ;
    DoubleStore(GaugeGrid(), Umu, HUmu);
    pickCheckerboard(Even, UmuEven, Umu);
    pickCheckerboard(Odd, UmuOdd, Umu);
  }

  inline void DoubleStore(GridBase *GaugeGrid,
                          DoubledGaugeField &Uds,
                          const GaugeField &Umu)
  {
  // copied from WilsonImpl
    typedef typename Simd::scalar_type scalar_type;

    conformable(Uds.Grid(), GaugeGrid);
    conformable(Umu.Grid(), GaugeGrid);

    GaugeLinkField U(GaugeGrid);
    GaugeLinkField tmp(GaugeGrid);

    Lattice<iScalar<vInteger> > coor(GaugeGrid);
    ////////////////////////////////////////////////////
    // apply any boundary phase or twists
    ////////////////////////////////////////////////////
    for (int mu : dimensions) {

      ////////// boundary phase /////////////
      auto pha = Impl::Params.boundary_phases[mu];
      scalar_type phase( real(pha),imag(pha) );

      int L   = GaugeGrid->GlobalDimensions()[mu];
      int Lmu = L - 1;

      LatticeCoordinate(coor, mu);

      U = PeekIndex<LorentzIndex>(Umu, mu);

      // apply any twists
      RealD theta = Impl::Params.twist_n_2pi_L[mu] * 2*M_PI / L;
      if ( theta != 0.0) {
        scalar_type twphase(::cos(theta),::sin(theta));
        U = twphase*U;
        std::cout << GridLogMessage << " Twist ["<<mu<<"] "<< Impl::Params.twist_n_2pi_L[mu]<< " phase"<<phase <<std::endl;
      }

      tmp = where(coor == Lmu, phase * U, U);
      PokeIndex<LorentzIndex>(Uds, tmp, mu);

      U = adj(Cshift(U, mu, -1));
      U = where(coor == 0, conjugate(phase) * U, U);
      PokeIndex<LorentzIndex>(Uds, U, mu + 4);
    }
  }

  // implement functionality
  void M(const FermionField& in, FermionField& out) {
    conformable(_grid,in.Grid());
    conformable(in.Grid(),out.Grid());
    out.Checkerboard() = in.Checkerboard();

    SimpleCompressor<SiteSpinor> compressor;
    Stencil.HaloExchange(in, compressor);

    autoView(in_v, in, AcceleratorRead);
    autoView(out_v, out, AcceleratorWrite);
    autoView(Stencil_v, Stencil, AcceleratorRead);
    autoView(gauge_v, Umu, AcceleratorRead);

    typedef decltype(coalescedRead(in_v[0])) calcSpinor;

    accelerator_for(osite, _grid->oSites(), FermionField::vector_object::Nsimd(), {
      int permute_type;
      StencilEntry* SE;
      calcSpinor result = -1.0 * npoint * coalescedRead(in_v[osite]);
      calcSpinor displaced;

      // for each point in stencil
      for (int point = 0; point < npoint; point++){
        SE = Stencil_v.GetEntry(permute_type, point, osite);

        if (SE->_is_local){
          displaced = coalescedReadPermute(in_v[SE->_offset], permute_type, SE->_permute);
        } else {
          displaced = coalescedRead(Stencil_v.CommBuf()[SE->_offset]);
        }

        // multiply link and add to result
        mac(&result(), &gauge_v[osite](dims_gauge[point]), &displaced());
      }
      coalescedWrite(out_v[osite], result);
    })
  }

  void Mdag(const FermionField& in, FermionField& out) {
    assert(0 && "TODO: implement this");
  }
  void Mdir(const FermionField& in, FermionField& out, int dir, int disp) {
    DhopDir(in, out, dir, disp);
  }
  void MdirAll(const FermionField& in, std::vector<FermionField>& out) {
    assert(0 && "TODO: implement this");
  }
  void Meooe(const FermionField &in, FermionField &out) {
    if (in.Checkerboard() == Odd) {
      DhopEO(in, out, DaggerNo);
    } else {
      DhopOE(in, out, DaggerNo);
    }
  }
  void MeooeDag(const FermionField& in, FermionField& out) {
    if (in.Checkerboard() == Odd) {
      DhopEO(in, out, DaggerYes);
    } else {
      DhopOE(in, out, DaggerYes);
    }
  }
  void Mooee(const FermionField& in, FermionField& out) {
    out.Checkerboard() = in.Checkerboard();
    typename FermionField::scalar_type scal(prefactor);
    out = scal * in;
  }
  void MooeeDag(const FermionField& in, FermionField& out) {
    out.Checkerboard() = in.Checkerboard();
    Mooee(in, out);
  }
  void MooeeInv(const FermionField& in, FermionField& out) {
    out.Checkerboard() = in.Checkerboard();
    out = (1.0/(prefactor))*in;
  }
  void MooeeInvDag(const FermionField& in, FermionField& out) {
    out.Checkerboard() = in.Checkerboard();
    MooeeInv(in,out);
  }
  void Dhop(const FermionField& in, FermionField& out, int dag) {
    conformable(in.Grid(), _grid);  // verifies full grid
    conformable(in.Grid(), out.Grid());

    out.Checkerboard() = in.Checkerboard();
    //DhopInternal(Stencil, Umu, in, out, dag);
  }
  void DhopOE(const FermionField& in, FermionField& out, int dag) {
    conformable(in.Grid(), _cbgrid);    // verifies half grid
    conformable(in.Grid(), out.Grid());  // drops the cb check

    assert(in.Checkerboard() == Even);
    out.Checkerboard() = Odd;

    //DhopInternal(StencilEven, UmuOdd, in, out, dag);
  }
  void DhopEO(const FermionField& in, FermionField& out, int dag) {
    conformable(in.Grid(), _cbgrid);    // verifies half grid
    conformable(in.Grid(), out.Grid());  // drops the cb check

    assert(in.Checkerboard() == Odd);
    out.Checkerboard() = Even;

    //DhopInternal(StencilOdd, UmuEven, in, out, dag);
  }
  void DhopDir(const FermionField& in, FermionField& out, int dir, int disp) {
    assert(0 && "TODO: implement this");
  }
  void DhopDeriv(GaugeField& mat, const FermionField& U, const FermionField& V, int dag) {
    assert(0 && "TODO: implement this");
  }
  void DhopDerivOE(GaugeField& mat, const FermionField& U, const FermionField& V, int dag) {
    assert(0 && "TODO: implement this");
  }
  void DhopDerivEO(GaugeField& mat, const FermionField& U, const FermionField& V, int dag) {
    assert(0 && "TODO: implement this");
  }

  /*void DhopInternal(StencilImpl& st, DoubledGaugeField& U, const FermionField& in, FermionField& out, int dag) {
    assert(0 && "TODO: implement this");
#if 0 // taken from WilsonfermionImplementation.h
    Compressor compressor(dag);
    st.HaloExchange(in, compressor);
    int Opt = WilsonKernelsStatic::Opt;
    if (dag == DaggerYes) {
      DhopDagKernel(Opt,st,U,st.CommBuf(),1,U.oSites(),in,out);
    } else {
      DhopKernel(Opt,st,U,st.CommBuf(),1,U.oSites(),in,out);
    }
#endif
  }*/

private:
  GridBase *_grid;
  GridBase *_cbgrid;

  FermionField _tmp;

  std::vector<int> dimensions;
  std::vector<int> dims_gauge;
  std::vector<int> directions;
  std::vector<int> displacements;
  int npoint;

  DoubledGaugeField Umu;
  DoubledGaugeField UmuEven;
  DoubledGaugeField UmuOdd;

  CartesianStencil<SiteSpinor, SiteSpinor, int> Stencil;
  CartesianStencil<SiteSpinor, SiteSpinor, int> StencilEven;
  CartesianStencil<SiteSpinor, SiteSpinor, int> StencilOdd;

  RealD prefactor = -2.0;
};

