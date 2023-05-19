/*
    GPT - Grid Python Toolkit
    Copyright (C) 2020  Daniel Richtmann (daniel.richtmann@ur.de, https://github.com/lehner/gpt)
                  2020  Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)

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

// NOTE when hermitianOverall is not given, this construction assumes the coarse links it gets passed to be constructed from a gamma5-hermitian Dirac operator -> watch out twisted mass people! (c.f. Mike Creutz in https://quark.phy.bnl.gov/~creutz/natal/lecture3.pdf)
// then the following relation (c.f. Matthias Rottmann's PHD thesis) holds
//
//   D_{A_{p,\tau},A_{q,\kappa}} = (-1)^(\tau+kappa) D_{A_{q,\kappa},A_{p,\tau}}
//
// where D is the coarse link between lattice sites p and q, and the A_ are submatrices in coarse color space corresponding to the coarse spin dofs \tau and \kappa
//
// in different nomenclature, this means
//
// - for Uc_mu(x) = [ A B ]      (i.e., the coarse spin dof made explicit, sizeof A,B,C,D is nbasis/2 x nbasis/2)
//                  [ C D ]
//
//     Uc_mu(x)^dag = [ A B ]^dag = [ A^dag C^dag ] = [  A' -B' ] = Uc_-mu(x+mu) * X, where X = (-1)^(\tau+kappa)
//                    [ C D ]       [ B^dag D^dag ]   [ -C'  D' ]
//
// - for Uc_self(x) = [ A B ]
//                    [ C D ]
//
//     Uc_self(x)^dag = [ A B ]^dag = [ A^dag C^dag ] = [  A' -B' ] = Uc_self(x) * X, where X = (-1)^(\tau+kappa)
//                      [ C D ]       [ B^dag D^dag ]   [ -C'  D' ]

// The purpose of this file is to make the coarse grid operator aware of virtual fields.
// The first implementation we pursued in gpt looped over instances of Grid's CoarsenedMatrix.
// This showed bad performance due to quadratic scaling of the number of comms with the number of virtual fields and also low computational perfomance on GPUs due to less parallelism within a single kernel call.
//
// To adress the first of these problems, we do the following:
// We want to perform only ONE communication per application of the operator. We resort to the following trick to achieve this:
// We copy the list of the 4d virtual fields into a temporary 5d object of the same 4d size and have the virtual index run in the 5th dimension.
// We then use Grid's stencil on this five-dimensional object and a strictly local 5th dimension.
// In our experiments we find that the overhead introduced by the copying is absolutely negligibly compared to communication time that would be necessary otherwise.

// To adress the second problem, we do the following:
// We pack every degree of freedom (index within a virtual field, list of virtual fields, number of rhs, sites) into the accelerator_for in order to ensure higher utilization of the GPU's compute capabilities.
// If we were to manage utilizing the GPU's shared memory we could improve upon this by another factor of 8 in parallelism.

// Note that we also provide a cpu version without parallelism over the index within a virtual field. This code is based on the observation that we saw a 30% performance increase on the a64fx


// Toggles
#define ROW_MAJOR // store virtual index of link fields in row major order, i.e., transpose them w.r.t. python ordering
// #define TENSOR_LAYOUT // have Lorentz index in tensor structure rather than std::vector
#define REFERENCE_SUMMATION_ORDER // keep summation order as in reference implementation


template<class Field>
void conformable(GridBase* grid, const PVector<Field>& field) {
  for(int v=0; v<field.size(); v++) {
    std::cout << GridLogDebug << "conformable(grid/pvec): starting check for index " << v << std::endl;
    conformable(grid, field[v].Grid());
    std::cout << GridLogDebug << "conformable(grid/pvec): ending   check for index " << v << std::endl;
  }
}


template<class Field>
void conformable(const PVector<Field>& lhs, const PVector<Field>& rhs) {
  assert(lhs.size() == rhs.size());
  for(int v=0; v<lhs.size(); v++) {
    std::cout << GridLogDebug << "conformable(pvec/pvec): starting check for index " << v << std::endl;
    conformable(lhs[v], rhs[v]);
    std::cout << GridLogDebug << "conformable(pvec/pvec):   ending check for index " << v << std::endl;
  }
}


template<class Field>
void constantCheckerboard(const PVector<Field>& in, PVector<Field>& out) {
  assert(in.size() == out.size());
  for(int v=0; v<in.size(); v++) {
    std::cout << GridLogDebug << "constantCheckerboard: start setting for index " << v << std::endl;
    out[v].Checkerboard() = in[v].Checkerboard();
    std::cout << GridLogDebug << "constantCheckerboard: end setting for index " << v << std::endl;
  }
}


template<class Field>
void changingCheckerboard(const PVector<Field>& in, PVector<Field>& out) {
  assert(in.size() == out.size());
  for(int v=0; v<in.size(); v++) {
    std::cout << GridLogDebug << "constantCheckerboard: start setting for index " << v << std::endl;
    if      (in[v].Checkerboard() == Even) out[v].Checkerboard() = Odd;
    else if (in[v].Checkerboard() == Odd)  out[v].Checkerboard() = Even;
    else assert(0);
    std::cout << GridLogDebug << "constantCheckerboard: end   setting for index " << v << std::endl;
  }
}


template<class Field>
bool checkerBoardIs(const PVector<Field>& in, int cb) {
  assert((cb==Odd) || (cb==Even));
  bool ret = true;
  for(int v=0; v<in.size(); v++) {
    ret = ret && in[v].Checkerboard() == cb;
  }
  return ret;
}


// this is the interface that gpt requires at the moment
template<class FermionField>
class MultiArgFermionOperatorBase {
public:
  /////////////////////////////////////////////////////////////////////////////
  //                                  grids                                  //
  /////////////////////////////////////////////////////////////////////////////
  typedef PVector<FermionField> VFermionField;

  virtual GridBase* FermionGrid()         = 0;
  virtual GridBase* FermionRedBlackGrid() = 0;
  virtual GridBase* GaugeGrid()           = 0;
  virtual GridBase* GaugeRedBlackGrid()   = 0;
  GridBase* Grid()         { return FermionGrid(); }
  GridBase* RedBlackGrid() { return FermionRedBlackGrid(); }

  /////////////////////////////////////////////////////////////////////////////
  //                         info about diagonal term                        //
  /////////////////////////////////////////////////////////////////////////////

  virtual int ConstEE()     = 0;
  virtual int isTrivialEE() = 0;

  /////////////////////////////////////////////////////////////////////////////
  //                            full cb operations                           //
  /////////////////////////////////////////////////////////////////////////////

  virtual void M(const VFermionField& in, VFermionField& out) = 0;
  virtual void Mdag(const VFermionField& in, VFermionField& out) = 0;
  virtual void Mdiag(const VFermionField& in, VFermionField& out) { Mooee(in, out); }
  virtual void Mdir(const VFermionField& in, VFermionField& out, int dir, int disp) = 0;

  /////////////////////////////////////////////////////////////////////////////
  //                            half cb operations                           //
  /////////////////////////////////////////////////////////////////////////////

  virtual void Meooe(const VFermionField& in, VFermionField& out) = 0;
  virtual void MeooeDag(const VFermionField& in, VFermionField& out) = 0;
  virtual void Mooee(const VFermionField& in, VFermionField& out) = 0;
  virtual void MooeeDag(const VFermionField& in, VFermionField& out) = 0;
  virtual void MooeeInv(const VFermionField& in, VFermionField& out) = 0;
  virtual void MooeeInvDag(const VFermionField& in, VFermionField& out) = 0;

  /////////////////////////////////////////////////////////////////////////////
  //               non-hermitian hopping term; half cb or both               //
  /////////////////////////////////////////////////////////////////////////////

  virtual void Dhop(const VFermionField& in, VFermionField& out, int dag) = 0;
  virtual void DhopOE(const VFermionField& in, VFermionField& out, int dag) = 0;
  virtual void DhopEO(const VFermionField& in, VFermionField& out, int dag) = 0;
  virtual void DhopDir(const VFermionField& in, VFermionField& out, int dir, int disp) = 0;

  /////////////////////////////////////////////////////////////////////////////
  //                               dminus stuff                              //
  /////////////////////////////////////////////////////////////////////////////

  virtual void Dminus(const VFermionField& in, VFermionField& out) = 0;
  virtual void DminusDag(const VFermionField& in, VFermionField& out) = 0;

  /////////////////////////////////////////////////////////////////////////////
  //                              import/export                              //
  /////////////////////////////////////////////////////////////////////////////

  virtual void ImportPhysicalFermionSource(const VFermionField& input, VFermionField& imported) = 0;
  virtual void ImportUnphysicalFermion(const VFermionField& input, VFermionField& imported) = 0;
  virtual void ExportPhysicalFermionSolution(const VFermionField& solution, VFermionField& exported) = 0;
  virtual void ExportPhysicalFermionSource(const VFermionField& solution, VFermionField& exported) = 0;
};

template<class CComplex,int NbasisVirtual>
class MultiArgVirtualCoarsenedMatrix : public MultiArgFermionOperatorBase<Lattice<iVector<CComplex,NbasisVirtual>>> {
public: // type definitions ///////////////////////////////////////////////////

  // site-wise types
  typedef         iVector<CComplex, NbasisVirtual>          SiteSpinor;
  typedef         iMatrix<CComplex, NbasisVirtual>          SiteMatrix;
  typedef iVector<iMatrix<CComplex, NbasisVirtual>, 2*Nd+1> DoubleStoredSiteMatrix;

  // lattice types = virtual fields
  typedef Lattice<SiteSpinor>             FermionField;
  typedef Lattice<SiteMatrix>             LinkField;
  typedef Lattice<SiteMatrix>             GridLayoutGaugeField;
  typedef Lattice<DoubleStoredSiteMatrix> DoubleStoredGaugeField;

  // choose gauge type depending on compilation
#if defined (TENSOR_LAYOUT)
  typedef DoubleStoredGaugeField  GaugeField;
#else
  typedef GridLayoutGaugeField    GaugeField;
#endif

  // physical fields, used internally
  typedef std::vector<FermionField>        PhysicalFermionField;
  typedef std::vector<LinkField>           PhysicalLinkField;
  typedef std::vector<GaugeField>          PhysicalGaugeField;

  // used by the outside world
  typedef CartesianStencil<SiteSpinor,SiteSpinor,DefaultImplParams> Stencil;
  typedef typename SiteSpinor::vector_type            vCoeff_t;
  typedef PVector<FermionField>                       VFermionField;
  typedef PVector<LinkField>                          VLinkField;
  typedef PVector<GaugeField>                         VGaugeField;

private: // member data ///////////////////////////////////////////////////////

  Geometry geom_;
  Geometry geomMultiArg_;

  GridBase* grid_;
  GridBase* cbGrid_;
  GridBase* gridMultiArg_;
  GridBase* cbGridMultiArg_;

  bool hermitianOverall_;

  uint64_t link_n_virtual_;
  uint64_t fermion_n_virtual_;
  uint64_t n_arg_;
  uint64_t nbasis_global_;

  Stencil stencilMultiArg_;
  Stencil stencilEvenMultiArg_;
  Stencil stencilOddMultiArg_;

  PhysicalGaugeField Uc_;
  PhysicalGaugeField UcEven_;
  PhysicalGaugeField UcOdd_;

  PhysicalLinkField UcSelf_;
  PhysicalLinkField UcSelfEven_;
  PhysicalLinkField UcSelfOdd_;

  PhysicalLinkField UcSelfInv_;
  PhysicalLinkField UcSelfInvEven_;
  PhysicalLinkField UcSelfInvOdd_;

  FermionField tmpMultiArg_;
  FermionField tmpEvenMultiArg_;
  FermionField tmpOddMultiArg_;

  Vector<RealD> dag_factor_;

  double MCalls;
  double MMiscTime;
  double MViewTime;
  double MView2Time;
  double MCopyTime;
  double MCommTime;
  double MComputeTime;
  double MTotalTime;

public: // member functions (implementing interface) //////////////////////////

  // grids
  GridBase* FermionGrid()         { return grid_; }
  GridBase* FermionRedBlackGrid() { return cbGrid_; }
  GridBase* GaugeGrid()           { return grid_; }
  GridBase* GaugeRedBlackGrid()   { return cbGrid_; }

  // info about diagonal term
  int ConstEE()     { return 0; }
  int isTrivialEE() { return 0; }

  //
  // For multi RHS, we should have a variable n_virtual set in the constructor; this can be inferred from the links?
  //
  
  // full cb operations
  virtual void M(const VFermionField& in, VFermionField& out) {
    MInternal(in, out);
  }
  virtual void Mdag(const VFermionField& in, VFermionField& out) {
    MdagInternal(in, out);
  }
  void Mdiag(const VFermionField& in, VFermionField& out) {
    Mooee(in, out);
  }
  void Mdir(const VFermionField& in, VFermionField& out, int dir, int disp) {
    DhopDir(in, out, dir, disp);
  }

  // half cb operations
  void Meooe(const VFermionField& in, VFermionField& out) {
    if(in[0].Checkerboard() == Odd) {
      DhopEO(in, out, DaggerNo);
    } else {
      DhopOE(in, out, DaggerNo);
    }
  }
  void MeooeDag(const VFermionField& in, VFermionField& out) {
    if(in[0].Checkerboard() == Odd) {
      DhopEO(in, out, DaggerYes);
    } else {
      DhopOE(in, out, DaggerYes);
    }
  }
  void Mooee(const VFermionField& in, VFermionField& out) {
    if(in[0].Grid()->_isCheckerBoarded) {
      if(in[0].Checkerboard() == Odd) {
        MooeeInternal(UcSelfOdd_, in, out, DaggerNo);
      } else {
        MooeeInternal(UcSelfEven_, in, out, DaggerNo);
      }
    } else {
      MooeeInternal(UcSelf_, in, out, DaggerNo);
    }
  }
  void MooeeDag(const VFermionField& in, VFermionField& out) {
    if(in[0].Grid()->_isCheckerBoarded) {
      if(in[0].Checkerboard() == Odd) {
        MooeeInternal(UcSelfOdd_, in, out, DaggerYes);
      } else {
        MooeeInternal(UcSelfEven_, in, out, DaggerYes);
      }
    } else {
      MooeeInternal(UcSelf_, in, out, DaggerYes);
    }
  }
  void MooeeInv(const VFermionField& in, VFermionField& out) {
    if(in[0].Grid()->_isCheckerBoarded) {
      if(in[0].Checkerboard() == Odd) {
        MooeeInternal(UcSelfInvOdd_, in, out, DaggerNo);
      } else {
        MooeeInternal(UcSelfInvEven_, in, out, DaggerNo);
      }
    } else {
      MooeeInternal(UcSelfInv_, in, out, DaggerNo);
    }
  }
  void MooeeInvDag(const VFermionField& in, VFermionField& out) {
    if(in[0].Grid()->_isCheckerBoarded) {
      if(in[0].Checkerboard() == Odd) {
        MooeeInternal(UcSelfInvOdd_, in, out, DaggerYes);
      } else {
        MooeeInternal(UcSelfInvEven_, in, out, DaggerYes);
      }
    } else {
      MooeeInternal(UcSelfInv_, in, out, DaggerYes);
    }
  }

  // non-hermitian hopping term; half cb or both
  void Dhop(const VFermionField& in, VFermionField& out, int dag) {
    conformable(grid_, in); // verifies full grid
    conformable(in[0].Grid(), out);

    constantCheckerboard(in, out);

    DhopInternal(stencilMultiArg_, Uc_, in, out, tmpMultiArg_, dag);
  }
  void DhopOE(const VFermionField& in, VFermionField& out, int dag) {
    conformable(cbGrid_, in);       // verifies half grid
    conformable(in[0].Grid(), out); // drops the cb check

    assert(checkerBoardIs(in, Even));
    changingCheckerboard(in, out);

    DhopInternal(stencilEvenMultiArg_, UcOdd_, in, out, tmpEvenMultiArg_, dag);
  }
  void DhopEO(const VFermionField& in, VFermionField& out, int dag) {
    conformable(cbGrid_, in);       // verifies half grid
    conformable(in[0].Grid(), out); // drops the cb check

    assert(checkerBoardIs(in, Odd));
    changingCheckerboard(in, out);

    DhopInternal(stencilOddMultiArg_, UcEven_, in, out, tmpOddMultiArg_, dag);
  }
  void DhopDir(const VFermionField& in, VFermionField& out, int dir, int disp) {
    conformable(grid_, in); // verifies full grid
    conformable(in[0].Grid(), out);

    constantCheckerboard(in, out);

    DhopDirInternal(stencilMultiArg_, Uc_, in, out, tmpMultiArg_, geom_.point(dir, disp));
  }

  // dminus stuff
  void Dminus(const VFermionField& in, VFermionField& out) {
    assert(0 && "TODO: implement this");
  }
  void DminusDag(const VFermionField& in, VFermionField& out) {
    assert(0 && "TODO: implement this");
  }

  // import/export
  void ImportPhysicalFermionSource(const VFermionField& input, VFermionField& imported) {
    assert(0 && "TODO: implement this");
  }
  void ImportUnphysicalFermion(const VFermionField& input, VFermionField& imported) {
    assert(0 && "TODO: implement this");
  }
  void ExportPhysicalFermionSolution(const VFermionField& solution, VFermionField& exported) {
    assert(0 && "TODO: implement this");
  }
  void ExportPhysicalFermionSource(const VFermionField& solution, VFermionField& exported) {
    assert(0 && "TODO: implement this");
  }

public: // member functions (additional) //////////////////////////////////////

  // helpers
  void ImportGauge(const PVector<LinkField>& Uc, const PVector<LinkField>& UcSelfInv) {
    const int Nsite           = GaugeGrid()->oSites();
    const int Nsimd           = SiteSpinor::Nsimd();
    const int Npoint          = geom_.npoint;
    const int NvirtualLink    = link_n_virtual_;
    const int NvirtualFermion = fermion_n_virtual_;

#if defined (TENSOR_LAYOUT) // Lorentz in tensor
    assert(Uc.size()        == Uc_.size() * geom_.npoint);
    assert(UcSelfInv.size() == UcSelfInv_.size());

    conformable(GaugeGrid(), Uc);
    conformable(GaugeGrid(), UcSelfInv);

    // NOTE: can't use PokeIndex here because of different tensor depths
    VECTOR_VIEW_OPEN(Uc_, Uc_member_v, AcceleratorWrite);
    VECTOR_VIEW_OPEN(Uc,  Uc_arg_v,    AcceleratorRead);
    for(int v=0; v<NvirtualLink; v++) {
      const int v_row = v%NvirtualFermion; const int v_col = v/NvirtualFermion; // NOTE: comes in from gpt with row faster index -> col-major order
#if defined(ROW_MAJOR)
      const int v_link = v_row * NvirtualFermion + v_col; // change to col faster index -> row-major order -> with    transpose
#else // =  COL_MAJOR
      const int v_link = v_col * NvirtualFermion + v_row; // keep      row faster index -> col-major order -> without transpose
#endif
      // new layout with Lorentz in tensor
      for(int p=0; p<Npoint; p++) {
        accelerator_forNB(ss, Nsite, Nsimd, {
          coalescedWrite(Uc_member_v[v_link][ss](p), coalescedRead(Uc_arg_v[p*NvirtualLink+v][ss]));
        });
      }
      accelerator_barrier();
    }
    VECTOR_VIEW_CLOSE(Uc_member_v);
    VECTOR_VIEW_CLOSE(Uc_arg_v);
#else // grid's layout with Lorentz in std::vector
    assert(Uc.size()        == Uc_.size());
    assert(UcSelfInv.size() == UcSelfInv_.size());

    conformable(GaugeGrid(), Uc);
    conformable(GaugeGrid(), UcSelfInv);

    VECTOR_VIEW_OPEN(Uc_, Uc_member_v, AcceleratorWrite);
    VECTOR_VIEW_OPEN(Uc,  Uc_arg_v,    AcceleratorRead);
    for(int v=0; v<NvirtualLink; v++) {
      const int v_row = v%NvirtualFermion; const int v_col = v/NvirtualFermion; // NOTE: comes in from gpt with row faster index -> col-major order

      for(int p=0; p<Npoint; p++) {
#if defined(ROW_MAJOR)
        const int v_point_link = p * NvirtualLink + v_row * NvirtualFermion + v_col;    // point slow, virtual fast, col faster index -> virtual in row-major order
        const int v_link_point = v_row * NvirtualFermion * Npoint + v_col * Npoint + p; // virtual slow, point fast, col faster index -> virtual in row-major order
#else // =  COL_MAJOR
        const int v_point_link = p * NvirtualLink + v_col * NvirtualFermion + v_row;    // point slow, virtual fast, row faster index -> virtual in col-major order
        const int v_link_point = v_col * NvirtualFermion * Npoint + v_row * Npoint + p; // virtual slow, point fast, row faster index -> virtual in col-major order
#endif

        // // these are the old indices
        // const int gauge_idx_point_row_col = p * NvirtualLink + v_row * NvirtualFermion + v_col;
        // const int gauge_idx_point_col_row = p * NvirtualLink + v_col * NvirtualFermion + v_row;
        // const int gauge_idx_row_col_point = v_row * NvirtualFermion * Npoint + v_col * Npoint + p;
        // const int gauge_idx_col_row_point = v_col * NvirtualFermion * Npoint + v_row * Npoint + p;

        accelerator_forNB(ss, Nsite, Nsimd, {
          // coalescedWrite(Uc_member_p[v_point_link][ss], coalescedRead(Uc_arg_p[p*NvirtualLink+v][ss]));
          coalescedWrite(Uc_member_v[v_link_point][ss], coalescedRead(Uc_arg_v[p*NvirtualLink+v][ss]));
        });
      }
    }
    accelerator_barrier();
    VECTOR_VIEW_CLOSE(Uc_member_v);
    VECTOR_VIEW_CLOSE(Uc_arg_v);
#endif

    auto self_point = geom_.npoint-1;
    for(int v=0; v<NvirtualLink; v++) {
      const int v_row = v%NvirtualFermion; const int v_col = v/NvirtualFermion; // NOTE: comes in from gpt with row faster index -> col-major order
#if defined(ROW_MAJOR)
      const int v_link = v_row * NvirtualFermion + v_col; // change to col faster index -> row-major order -> with    transpose
#else // =  COL_MAJOR
      const int v_link = v_col * NvirtualFermion + v_row; // keep      row faster index -> col-major order -> without transpose
#endif

      UcSelf_[v_link]    = Uc[self_point*NvirtualLink+v];
      UcSelfInv_[v_link] = UcSelfInv[v];
    }

  }

  void PickCheckerboards() {
#if defined (TENSOR_LAYOUT)
    assert(Uc_.size()        == link_n_virtual_);
#else
    assert(Uc_.size()        == link_n_virtual_ * geom_.npoint);
#endif
    assert(UcSelf_.size()    == link_n_virtual_);
    assert(UcSelfInv_.size() == link_n_virtual_);

    assert(UcEven_.size() == Uc_.size());
    assert(UcOdd_.size()  == Uc_.size());
    assert(UcSelfEven_.size() == UcSelf_.size());
    assert(UcSelfOdd_.size()  == UcSelf_.size());
    assert(UcSelfInvEven_.size() == UcSelfInv_.size());
    assert(UcSelfInvOdd_.size()  == UcSelfInv_.size());

    for(int i=0; i<Uc_.size(); i++) {
      pickCheckerboard(Even, UcEven_[i], Uc_[i]);
      pickCheckerboard(Odd,   UcOdd_[i], Uc_[i]);
    }

    for(int i=0; i<UcSelf_.size(); i++) {
      pickCheckerboard(Even, UcSelfEven_[i], UcSelf_[i]);
      pickCheckerboard(Odd, UcSelfOdd_[i], UcSelf_[i]);
    }

    for(int i=0; i<UcSelfInv_.size(); i++) {
      pickCheckerboard(Even, UcSelfInvEven_[i], UcSelfInv_[i]);
      pickCheckerboard(Odd, UcSelfInvOdd_[i], UcSelfInv_[i]);
    }

  }

  void Report(int Nvec) {
    /*
    assert(Nvec == n_arg_);
    RealD Nproc = grid_->_Nprocessors;
    RealD Nnode = grid_->NodeCount();
    RealD volume = 1;
    Coordinate latt = grid_->GlobalDimensions();
    for(int mu=0;mu<Nd;mu++) volume=volume*latt[mu];
    RealD nbasis = NbasisVirtual * fermion_n_virtual_;

    if ( MCalls > 0 ) {
      grid_message("#### M calls report\n");
      grid_message("CoarseOperator Number of Calls                         : %d\n", (int)MCalls);
      grid_message("CoarseOperator MiscTime   /Calls, MiscTime    : %10.2f us, %10.2f us (= %6.2f %%)\n", MMiscTime   /MCalls, MMiscTime,    MMiscTime   /MTotalTime*100);
      grid_message("CoarseOperator ViewTime   /Calls, ViewTime    : %10.2f us, %10.2f us (= %6.2f %%)\n", MViewTime   /MCalls, MViewTime,    MViewTime   /MTotalTime*100);
      grid_message("CoarseOperator View2Time  /Calls, View2Time   : %10.2f us, %10.2f us (= %6.2f %%)\n", MView2Time  /MCalls, MView2Time,   MView2Time  /MTotalTime*100);
      grid_message("CoarseOperator CopyTime   /Calls, CopyTime    : %10.2f us, %10.2f us (= %6.2f %%)\n", MCopyTime   /MCalls, MCopyTime,    MCopyTime   /MTotalTime*100);
      grid_message("CoarseOperator CommTime   /Calls, CommTime    : %10.2f us, %10.2f us (= %6.2f %%)\n", MCommTime   /MCalls, MCommTime,    MCommTime   /MTotalTime*100);
      grid_message("CoarseOperator ComputeTime/Calls, ComputeTime : %10.2f us, %10.2f us (= %6.2f %%)\n", MComputeTime/MCalls, MComputeTime, MComputeTime/MTotalTime*100);
      grid_message("CoarseOperator TotalTime  /Calls, TotalTime   : %10.2f us, %10.2f us (= %6.2f %%)\n", MTotalTime  /MCalls, MTotalTime,   MTotalTime  /MTotalTime*100);
      
      // Average the compute time
      grid_->GlobalSum(MComputeTime);
      MComputeTime/=Nproc;
      RealD complex_words = 2;
      RealD prec_bytes    = getPrecision<typename CComplex::vector_type>::value * 4; // 4 for float, 8 for double
      RealD flop_per_site = 1.0 * (2 * nbasis * (36 * nbasis - 1)) * Nvec;
      RealD word_per_site = 1.0 * (9 * nbasis + 9 * nbasis * nbasis + nbasis) * Nvec;
      RealD byte_per_site = word_per_site * complex_words * prec_bytes;
      RealD mflops = flop_per_site*volume*MCalls/MComputeTime;
      RealD mbytes = byte_per_site*volume*MCalls/MComputeTime;
      grid_message("CoarseOperator Average mflops/s, mbytes/s per call                : %.0f, %.0f\n", mflops, mbytes);
      grid_message("CoarseOperator Average mflops/s, mbytes/s per call per rank       : %.0f, %.0f\n", mflops/Nproc, mbytes/Nproc);
      grid_message("CoarseOperator Average mflops/s, mbytes/s per call per node       : %.0f, %.0f\n", mflops/Nnode, mbytes/Nnode);

      RealD Fullmflops = flop_per_site*volume*MCalls/(MTotalTime);
      RealD Fullmbytes = byte_per_site*volume*MCalls/(MTotalTime);
      grid_message("CoarseOperator Average mflops/s, mbytes/s per call (full)         : %.0f, %.0f\n", Fullmflops, Fullmbytes);
      grid_message("CoarseOperator Average mflops/s, mbytes/s per call per rank (full): %.0f, %.0f\n", Fullmflops/Nproc, Fullmbytes/Nproc);
      grid_message("CoarseOperator Average mflops/s, mbytes/s per call per node (full): %.0f, %.0f\n", Fullmflops/Nnode, Fullmbytes/Nnode);

      grid_message("CoarseOperator StencilMultiArg\n"); stencilMultiArg_.Report();
      grid_message("CoarseOperator StencilMultiArgEven\n"); stencilMultiArgEven_.Report();
      grid_message("CoarseOperator StencilMultiArgOdd\n"); stencilMultiArgOdd_.Report();
    }
    grid_message("Report of new Coarse Operator finished\n");
    */
  }

  void ZeroCounters() {
    MCalls       = 0; // ok
    MMiscTime    = 0;
    MViewTime    = 0;
    MView2Time   = 0;
    MCopyTime    = 0;
    MCommTime    = 0;
    MComputeTime = 0;
    MTotalTime   = 0;

    //stencilMultiArg_.ZeroCounters();
    //stencilEvenMultiArg_.ZeroCounters();
    //stencilOddMultiArg_.ZeroCounters();
  }

  void copyToTmp5dField(const VFermionField& in, FermionField& tmp, int Nsite, int NvirtualFermion, int Narg) {
    MView2Time-=usecond();
    VECTOR_VIEW_OPEN(in, in_v, AcceleratorRead);
    autoView(tmp_v, tmp, AcceleratorWrite);
    auto tmp_p = &tmp_v[0];
    MView2Time+=usecond();
    accelerator_for(sF, Nsite*NvirtualFermion*Narg, SiteSpinor::Nsimd(), {
        int _sF   = sF;                  // this does fastest to slowest from top to bottom
        const int arg   = _sF%Narg;            _sF/=Narg;
        const int v_col = _sF%NvirtualFermion; _sF/=NvirtualFermion;
        const int sU    = _sF%Nsite;           _sF/=Nsite;
        coalescedWrite(tmp_p[sF], coalescedRead(in_v[arg*NvirtualFermion+v_col][sU]));
        // grid_message("COPY: sF = %4d, arg = %4d, sU = %4d, v_col = %4d\n", sF, arg, sU, v_col); fflush(stdout);
      });
    MCopyTime+=usecond();
    MView2Time-=usecond();
    VECTOR_VIEW_CLOSE(in_v);
    MView2Time+=usecond();
  }

  // constructors
  MultiArgVirtualCoarsenedMatrix(const PVector<LinkField>& Uc,
                                 const PVector<LinkField>& UcSelfInv,
                                 GridCartesian&                 grid,
                                 GridRedBlackCartesian&         rbGrid,
                                 int                            makeHermitian,
                                 int                            numArg)
    : geom_(grid._ndimension)
    , geomMultiArg_(grid._ndimension+1)
    , grid_(&grid)
    , cbGrid_(&rbGrid)
    , gridMultiArg_(SpaceTimeGrid::makeFiveDimGrid(numArg*uint64_t(sqrt(UcSelfInv.size())), &grid))
    , cbGridMultiArg_(SpaceTimeGrid::makeFiveDimRedBlackGrid(numArg*uint64_t(sqrt(UcSelfInv.size())), &grid))
    , hermitianOverall_(makeHermitian)
    , link_n_virtual_(UcSelfInv.size())
    , fermion_n_virtual_(uint64_t(sqrt(UcSelfInv.size())))
    , n_arg_(numArg)
    , nbasis_global_(fermion_n_virtual_*NbasisVirtual)
    , stencilMultiArg_(gridMultiArg_, geomMultiArg_.npoint, Even, geomMultiArg_.directions, geomMultiArg_.displacements)
    , stencilEvenMultiArg_(cbGridMultiArg_, geomMultiArg_.npoint, Even, geomMultiArg_.directions, geomMultiArg_.displacements)
    , stencilOddMultiArg_(cbGridMultiArg_, geomMultiArg_.npoint, Odd, geomMultiArg_.directions, geomMultiArg_.displacements)
#if 0 // for separating the self-stencil from the directions
    , stencilMultiArg_(gridMultiArg_, geomMultiArg_.npoint-1, Even, geomMultiArg_.directions, geomMultiArg_.displacements)
    , stencilEvenMultiArg_(cbGridMultiArg_, geomMultiArg_.npoint-1, Even, geomMultiArg_.directions, geomMultiArg_.displacements)
    , stencilOddMultiArg_(cbGridMultiArg_, geomMultiArg_.npoint-1, Odd, geomMultiArg_.directions, geomMultiArg_.displacements)
#endif
#if defined (TENSOR_LAYOUT)
    , Uc_(link_n_virtual_, grid_)
    , UcEven_(link_n_virtual_, cbGrid_)
    , UcOdd_(link_n_virtual_, cbGrid_)
#else
    , Uc_(geom_.npoint*link_n_virtual_, grid_)
    , UcEven_(geom_.npoint*link_n_virtual_, cbGrid_)
    , UcOdd_(geom_.npoint*link_n_virtual_, cbGrid_)
#endif
    , UcSelf_(link_n_virtual_, grid_)
    , UcSelfEven_(link_n_virtual_, cbGrid_)
    , UcSelfOdd_(link_n_virtual_, cbGrid_)
    , UcSelfInv_(link_n_virtual_, grid_)
    , UcSelfInvEven_(link_n_virtual_, cbGrid_)
    , UcSelfInvOdd_(link_n_virtual_, cbGrid_)
    , tmpMultiArg_(gridMultiArg_)
    , tmpEvenMultiArg_(cbGridMultiArg_)
    , tmpOddMultiArg_(cbGridMultiArg_)
    , dag_factor_(nbasis_global_*nbasis_global_)
  {
    //grid_->show_decomposition();
    //cbGrid_->show_decomposition();
    //gridMultiArg_->show_decomposition();
    //cbGridMultiArg_->show_decomposition();
    ImportGauge(Uc, UcSelfInv);
    PickCheckerboards();
    ZeroCounters();

    // need to set these here explicitly since I can't pick a cb for them!
    tmpEvenMultiArg_.Checkerboard() = Even;
    tmpOddMultiArg_.Checkerboard() = Odd;

    fillFactor();

    assert(n_arg_ == 1); // Limit to 1 for now!  This needs a re-work.
  }

private: // member functions //////////////////////////////////////////////////

  inline int numArg(const VFermionField& a, const VFermionField& b) const {
    int a_size = a.size();
    int b_size = b.size();
    assert(a_size == b_size);
    // TODO use operator size deduced from gauge fields
    return 1;
  }

  inline int numVirtual(const VFermionField& a, const VFermionField& b) const {
    return a.size();
  }

  void fillFactor() {
    Eigen::MatrixXd dag_factor_eigen = Eigen::MatrixXd::Ones(nbasis_global_, nbasis_global_);
    if(!hermitianOverall_) {
      const int nb = nbasis_global_/2;
      dag_factor_eigen.block(0,nb,nb,nb) *= -1.0;
      dag_factor_eigen.block(nb,0,nb,nb) *= -1.0;
    }

    // GPU readable prefactor
    thread_for(i, nbasis_global_*nbasis_global_, {
      int col = i%nbasis_global_;
      int row = i/nbasis_global_;
      dag_factor_[i] = dag_factor_eigen(row, col);
    });
  }

public: // kernel functions TODO: move somewhere else ////////////////////////

  void MInternal(const VFermionField& in, VFermionField& out) {
#ifdef GRID_HAS_ACCELERATOR
    MInternal_gpu(in, out);
#else
    MInternal_cpu(in, out);
#endif
  }

  void MdagInternal(const VFermionField& in, VFermionField& out) {
#ifdef GRID_HAS_ACCELERATOR
    MdagInternal_gpu(in, out);
#else
    MdagInternal_cpu(in, out);
#endif
  }

  void DhopInternal(Stencil& stencil, const PhysicalGaugeField& Uc, const VFermionField& in, VFermionField& out, FermionField& tmp, int dag) {
#ifdef GRID_HAS_ACCELERATOR
    if(dag == DaggerYes)
      DhopDagInternal_gpu(stencil, Uc, in, out, tmp);
    else
      DhopInternal_gpu(stencil, Uc, in, out, tmp);
#else
    if(dag == DaggerYes)
      DhopDagInternal_cpu(stencil, Uc, in, out, tmp);
    else
      DhopInternal_cpu(stencil, Uc, in, out, tmp);
#endif
  }

  void DhopDirInternal(Stencil& stencil, const PhysicalGaugeField& Uc, const VFermionField& in, VFermionField& out, FermionField& tmp, int point) {
#ifdef GRID_HAS_ACCELERATOR
    DhopDirInternal_gpu(stencil, Uc, in, out, tmp, point);
#else
    DhopDirInternal_cpu(stencil, Uc, in, out, tmp, point);
#endif
  }

  void MooeeInternal(const PhysicalLinkField& UcSelf, const VFermionField& in, VFermionField& out, int dag) {
#ifdef GRID_HAS_ACCELERATOR
    if(dag == DaggerYes)
      MooeeDagInternal_gpu(UcSelf, in, out);
    else
      MooeeInternal_gpu(UcSelf, in, out);
#else
    if(dag == DaggerYes)
      MooeeDagInternal_cpu(UcSelf, in, out);
    else
      MooeeInternal_cpu(UcSelf, in, out);
#endif
  }

  void MInternal_gpu(const VFermionField& in, VFermionField& out) {
    const int Narg            = numArg(in, out);
    const int NvirtualFermion = numVirtual(in, out);
    const int NvirtualLink    = NvirtualFermion*NvirtualFermion;
    const int Nsimd           = SiteSpinor::Nsimd();
    const int Nsite           = in[0].Grid()->oSites();
    const int Npoint          = geom_.npoint;

    assert(n_arg_ == Narg);

    conformable(FermionGrid(), in);
    conformable(FermionGrid(), out);
    constantCheckerboard(in, out);

    Timer("compressor");
    SimpleCompressor<SiteSpinor> compressor;


    Timer("copyToTmp5dField");
    copyToTmp5dField(in, tmpMultiArg_, Nsite, NvirtualFermion, Narg); // timed inside

    Timer("HaloExchange");
    stencilMultiArg_.HaloExchange(tmpMultiArg_, compressor);

    Timer("View");
    VECTOR_VIEW_OPEN(Uc_, Uc_v, AcceleratorRead);
    autoView(tmpMultiArg_v, tmpMultiArg_, AcceleratorRead);
    autoView(stencilMultiArg_v, stencilMultiArg_, AcceleratorRead);
    auto tmpMultiArg_p = &tmpMultiArg_v[0];
    VECTOR_VIEW_OPEN(out, out_v, AcceleratorWrite);


    Timer("Compute");
    for(int v_col=0; v_col<NvirtualFermion; v_col++) {
      typedef decltype(coalescedRead(tmpMultiArg_v[0]))    calcVector;
      typedef decltype(coalescedRead(tmpMultiArg_v[0](0))) calcComplex;

      /*
	Checked multiple streams and breaking this up in more NB, did not improve
      */
      accelerator_forNB(idx, Nsite*NvirtualFermion*Narg*NbasisVirtual, Nsimd, {
	  int _idx  = idx;
	  const int b     = _idx%NbasisVirtual; _idx/=NbasisVirtual;
	  const int arg   = _idx%Narg; _idx/=Narg;
	  const int v_row = _idx%NvirtualFermion; _idx/=NvirtualFermion;
	  const int ss    = _idx%Nsite; _idx/=Nsite;
	  
	  const int v_arg_col = arg*NvirtualFermion+v_col;
	  const int v_arg_row = arg*NvirtualFermion+v_row;
	  const int v_row_col = v_row * NvirtualFermion + v_col;
	  const int v_col_row = v_col * NvirtualFermion + v_row;
	  const int sF        = ss*NvirtualFermion*Narg+v_col*Narg+arg; // needed for stencil access
	  
#if defined(ROW_MAJOR)
	  const int v_link = v_row_col;
#else // =  COL_MAJOR
	  const int v_link = v_col_row;
#endif
	  
	  calcComplex res;
	  calcVector nbr;
	  int ptype;
	  StencilEntry *SE_MA;
	  
#if defined(REFERENCE_SUMMATION_ORDER)
	  res = Zero();
#else
	  if (v_col == 0)
	    res = Zero();
	  else
	    res = coalescedRead(out_v[v_arg_row][ss](b));
#endif
	  
	  for(int point=0; point<Npoint; point++) {
	    SE_MA=stencilMultiArg_v.GetEntry(ptype,point,sF);
	    
	    if(SE_MA->_is_local) {
	      nbr = coalescedReadPermute(tmpMultiArg_p[SE_MA->_offset],ptype,SE_MA->_permute);
	    } else {
	      nbr = coalescedRead(stencilMultiArg_v.CommBuf()[SE_MA->_offset]);
	    }
	    acceleratorSynchronise();
	    
	    for(int bb=0;bb<NbasisVirtual;bb++) {
#if defined(TENSOR_LAYOUT)
	      res = res + coalescedRead(Uc_v[v_link][ss](point)(b,bb))*nbr(bb);
#else
	      res = res + coalescedRead(Uc_v[v_link*Npoint+point][ss](b,bb))*nbr(bb);
#endif
	    }
	  }
#if defined(REFERENCE_SUMMATION_ORDER)
	  if (v_col != 0) {
	    res = res + coalescedRead(out_v[v_arg_row][ss](b));
	  }
#endif
	  coalescedWrite(out_v[v_arg_row][ss](b),res);
	});
    }
    accelerator_barrier();
    Timer("View");
    VECTOR_VIEW_CLOSE(Uc_v);
    VECTOR_VIEW_CLOSE(out_v);
    Timer();
  }

  void MInternal_cpu(const VFermionField& in, VFermionField& out) {
    MCalls++;
    MTotalTime -= usecond();
    MMiscTime -= usecond();
    const int Narg            = numArg(in, out);
    const int NvirtualFermion = numVirtual(in, out);
    const int NvirtualLink    = NvirtualFermion*NvirtualFermion;
    const int Nsimd           = SiteSpinor::Nsimd();
    const int Nsite           = in[0].Grid()->oSites();
    const int Npoint          = geom_.npoint;

    assert(n_arg_ == Narg);

    conformable(FermionGrid(), in);
    conformable(FermionGrid(), out);
    constantCheckerboard(in, out);

    SimpleCompressor<SiteSpinor> compressor;
    MMiscTime += usecond();

    copyToTmp5dField(in, tmpMultiArg_, Nsite, NvirtualFermion, Narg); // timed inside

    MCommTime-=usecond();
    stencilMultiArg_.HaloExchange(tmpMultiArg_, compressor);
    MCommTime+=usecond();

    MViewTime -= usecond();
    VECTOR_VIEW_OPEN(Uc_, Uc_v, AcceleratorRead);
    autoView(tmpMultiArg_v, tmpMultiArg_, AcceleratorRead);
    autoView(stencilMultiArg_v, stencilMultiArg_, AcceleratorRead);
    VECTOR_VIEW_OPEN(out, out_v, AcceleratorWrite);
    MViewTime += usecond();

    for(int v_col=0; v_col<NvirtualFermion; v_col++) {
      typedef decltype(coalescedRead(tmpMultiArg_v[0]))    calcVector;
      typedef decltype(coalescedRead(tmpMultiArg_v[0](0))) calcComplex;

      MComputeTime -= usecond();
      accelerator_for(idx, Nsite*NvirtualFermion*Narg, Nsimd, {
              int _idx  = idx;
        const int arg   = _idx%Narg; _idx/=Narg;
        const int v_row = _idx%NvirtualFermion; _idx/=NvirtualFermion;
        const int ss    = _idx%Nsite; _idx/=Nsite;

        const int v_arg_col = arg*NvirtualFermion+v_col;
        const int v_arg_row = arg*NvirtualFermion+v_row;
        const int v_row_col = v_row * NvirtualFermion + v_col;
        const int v_col_row = v_col * NvirtualFermion + v_row;
        const int sF        = ss*NvirtualFermion*Narg+v_col*Narg+arg; // needed for stencil access

#if defined(ROW_MAJOR)
        const int v_link = v_row_col;
#else // =  COL_MAJOR
        const int v_link = v_col_row;
#endif

        calcVector res;
        calcVector nbr;
        int ptype;
        StencilEntry *SE_MA;

#if defined(REFERENCE_SUMMATION_ORDER)
        res = Zero();
#else
        if (v_col == 0)
          res = Zero();
        else
          res = coalescedRead(out_v[v_arg_row][ss]);
#endif

        for(int point=0; point<Npoint; point++) {
          SE_MA=stencilMultiArg_v.GetEntry(ptype,point,sF);

          if(SE_MA->_is_local) {
            nbr = coalescedReadPermute(tmpMultiArg_v[SE_MA->_offset],ptype,SE_MA->_permute);
          } else {
            nbr = coalescedRead(stencilMultiArg_v.CommBuf()[SE_MA->_offset]);
          }
          acceleratorSynchronise();

#if defined(TENSOR_LAYOUT)
          res = res + coalescedRead(Uc_v[v_link][ss](point))*nbr;
#else
          res = res + coalescedRead(Uc_v[v_link*Npoint+point][ss])*nbr;
#endif
        }

#if defined(REFERENCE_SUMMATION_ORDER)
        if (v_col != 0) {
          res = res + coalescedRead(out_v[v_arg_row][ss]);
        }
#endif
        coalescedWrite(out_v[v_arg_row][ss],res);
      });
      MComputeTime += usecond();
    }
    MViewTime -= usecond();
    VECTOR_VIEW_CLOSE(Uc_v);
    VECTOR_VIEW_CLOSE(out_v);
    MViewTime += usecond();
    MTotalTime += usecond();
    //grid_message("Finished calling: MInternal_cpu\n");
  }

  void MdagInternal_gpu(const VFermionField& in, VFermionField& out) {
    MCalls++;
    MTotalTime -= usecond();
    MMiscTime -= usecond();
    const int Narg            = numArg(in, out);
    const int NvirtualFermion = numVirtual(in, out);
    const int NvirtualLink    = NvirtualFermion*NvirtualFermion;
    const int Nsimd           = SiteSpinor::Nsimd();
    const int Nsite           = in[0].Grid()->oSites();
    const int Npoint          = geom_.npoint;

    assert(n_arg_ == Narg);

    conformable(FermionGrid(), in);
    conformable(FermionGrid(), out);
    constantCheckerboard(in, out);

    SimpleCompressor<SiteSpinor> compressor;
    MMiscTime += usecond();

    copyToTmp5dField(in, tmpMultiArg_, Nsite, NvirtualFermion, Narg); // timed inside

    MCommTime-=usecond();
    stencilMultiArg_.HaloExchange(tmpMultiArg_, compressor);
    MCommTime+=usecond();

    MViewTime -= usecond();
    VECTOR_VIEW_OPEN(Uc_, Uc_v, AcceleratorRead);
    autoView(tmpMultiArg_v, tmpMultiArg_, AcceleratorRead);
    autoView(stencilMultiArg_v, stencilMultiArg_, AcceleratorRead);
    auto tmpMultiArg_p = &tmpMultiArg_v[0];
    VECTOR_VIEW_OPEN(out, out_v, AcceleratorWrite);
    RealD* dag_factor_p = &dag_factor_[0];
    auto nbasis_global__ = nbasis_global_; // auto-offloading problem of member function
    std::cout << GridLogMessage << dag_factor_.size() << " versus " << nbasis_global_ << std::endl;
    MViewTime += usecond();

    for(int v_col=0; v_col<NvirtualFermion; v_col++) {
      typedef decltype(coalescedRead(tmpMultiArg_v[0]))    calcVector;
      typedef decltype(coalescedRead(tmpMultiArg_v[0](0))) calcComplex;

      MComputeTime -= usecond();
      accelerator_forNB(idx, Nsite*NvirtualFermion*Narg*NbasisVirtual, Nsimd, {
	  int _idx  = idx;
	  const int b     = _idx%NbasisVirtual; _idx/=NbasisVirtual;
	  const int arg   = _idx%Narg; _idx/=Narg;
	  const int v_row = _idx%NvirtualFermion; _idx/=NvirtualFermion;
	  const int ss    = _idx%Nsite; _idx/=Nsite;
	  
	  const int v_arg_col = arg*NvirtualFermion+v_col;
	  const int v_arg_row = arg*NvirtualFermion+v_row;
	  const int v_row_col = v_row * NvirtualFermion + v_col;
	  const int v_col_row = v_col * NvirtualFermion + v_row;
	  const int sF        = ss*NvirtualFermion*Narg+v_col*Narg+arg; // needed for stencil access
	  
	  const int b_global = v_row*NvirtualFermion+b;
	  
#if defined(ROW_MAJOR)
	  const int v_link = v_row_col;
#else // =  COL_MAJOR
	  const int v_link = v_col_row;
#endif
	  
	  calcComplex res;
	  calcVector nbr;
	  int ptype;
	  StencilEntry *SE_MA;
	  
#if defined(REFERENCE_SUMMATION_ORDER)
	  res = Zero();
#else
	  if (v_col == 0)
	    res = Zero();
	  else
	    res = coalescedRead(out_v[v_arg_row][ss](b));
#endif
	  
	  for(int point=0; point<Npoint; point++) {
	    SE_MA=stencilMultiArg_v.GetEntry(ptype,point,sF);
	    
	    if(SE_MA->_is_local) {
	      nbr = coalescedReadPermute(tmpMultiArg_v[SE_MA->_offset],ptype,SE_MA->_permute);
	    } else {
	      nbr = coalescedRead(stencilMultiArg_v.CommBuf()[SE_MA->_offset]);
	    }
	    acceleratorSynchronise();

	    // suspect memory access problem in dag_factor_p of size n_basis_global_^2
	    for(int bb=0;bb<NbasisVirtual;bb++) {
	      const int bb_global = v_row*NvirtualFermion+bb;
#if defined(TENSOR_LAYOUT)
	      res = res + dag_factor_p[b_global*nbasis_global__+bb_global] * coalescedRead(Uc_v[v_link][ss](point)(b,bb))*nbr(bb);
#else
	      res = res + dag_factor_p[b_global*nbasis_global__+bb_global] * coalescedRead(Uc_v[v_link*Npoint+point][ss](b,bb))*nbr(bb);
#endif
	    }
	  }
#if defined(REFERENCE_SUMMATION_ORDER)
	  if (v_col != 0) {
	    res = res + coalescedRead(out_v[v_arg_row][ss](b));
	  }
#endif
	  coalescedWrite(out_v[v_arg_row][ss](b),res);
	});
      MComputeTime += usecond();
    }
    accelerator_barrier();
    MViewTime -= usecond();
    VECTOR_VIEW_CLOSE(Uc_v);
    VECTOR_VIEW_CLOSE(out_v);
    MViewTime += usecond();
    MTotalTime += usecond();
    //grid_message("Finished calling: MdagInternal_gpu\n");
  }

  void MdagInternal_cpu(const VFermionField& in, VFermionField& out) {
    MdagInternal_gpu(in, out);
  }

  void DhopInternal_gpu(Stencil& stencil, const PhysicalGaugeField& Uc, const VFermionField& in, VFermionField& out, FermionField& tmp) {
    MCalls++;
    MTotalTime -= usecond();
    MMiscTime -= usecond();
    const int Narg            = numArg(in, out);
    const int NvirtualFermion = numVirtual(in, out);
    const int NvirtualLink    = NvirtualFermion*NvirtualFermion;
    const int Nsimd           = SiteSpinor::Nsimd();
    const int Nsite           = in[0].Grid()->oSites();
    const int Npoint          = geom_.npoint;
    const int Npoint_hop      = geom_.npoint-1; // all but the self-coupling term

    assert(n_arg_ == Narg);

    SimpleCompressor<SiteSpinor> compressor;
    MMiscTime += usecond();

    copyToTmp5dField(in, tmp, Nsite, NvirtualFermion, Narg); // timed inside

    MCommTime-=usecond();
    stencil.HaloExchange(tmp, compressor);
    MCommTime+=usecond();

    MViewTime -= usecond();
    VECTOR_VIEW_OPEN(Uc, Uc_v, AcceleratorRead);
    autoView(tmp_v, tmp, AcceleratorRead);
    autoView(stencil_v, stencil, AcceleratorRead);
    auto tmp_p = &tmp_v[0];
    VECTOR_VIEW_OPEN(out, out_v, AcceleratorWrite);
    MViewTime += usecond();

    for(int v_col=0; v_col<NvirtualFermion; v_col++) {
      typedef decltype(coalescedRead(tmp_v[0]))    calcVector;
      typedef decltype(coalescedRead(tmp_v[0](0))) calcComplex;

      MComputeTime -= usecond();
      accelerator_forNB(idx, Nsite*NvirtualFermion*Narg*NbasisVirtual, Nsimd, {
              int _idx  = idx;
        const int b     = _idx%NbasisVirtual; _idx/=NbasisVirtual;
        const int arg   = _idx%Narg; _idx/=Narg;
        const int v_row = _idx%NvirtualFermion; _idx/=NvirtualFermion;
        const int ss    = _idx%Nsite; _idx/=Nsite;

        const int v_arg_col = arg*NvirtualFermion+v_col;
        const int v_arg_row = arg*NvirtualFermion+v_row;
        const int v_row_col = v_row * NvirtualFermion + v_col;
        const int v_col_row = v_col * NvirtualFermion + v_row;
        const int sF        = ss*NvirtualFermion*Narg+v_col*Narg+arg; // needed for stencil access

#if defined(ROW_MAJOR)
        const int v_link = v_row_col;
#else // =  COL_MAJOR
        const int v_link = v_col_row;
#endif

        calcComplex res;
        calcVector nbr;
        int ptype;
        StencilEntry *SE_MA;

#if defined(REFERENCE_SUMMATION_ORDER)
        res = Zero();
#else
        if (v_col == 0)
          res = Zero();
        else
          res = coalescedRead(out_v[v_arg_row][ss](b));
#endif

        for(int point=0; point<Npoint_hop; point++) {
          SE_MA=stencil_v.GetEntry(ptype,point,sF);

          if(SE_MA->_is_local) {
            nbr = coalescedReadPermute(tmp_p[SE_MA->_offset],ptype,SE_MA->_permute);
          } else {
            nbr = coalescedRead(stencil_v.CommBuf()[SE_MA->_offset]);
          }
          acceleratorSynchronise();

          for(int bb=0;bb<NbasisVirtual;bb++) {
#if defined(TENSOR_LAYOUT)
            res = res + coalescedRead(Uc_v[v_link][ss](point)(b,bb))*nbr(bb);
#else
            res = res + coalescedRead(Uc_v[v_link*Npoint+point][ss](b,bb))*nbr(bb);
#endif
          }
        }
#if defined(REFERENCE_SUMMATION_ORDER)
        if (v_col != 0) {
          res = res + coalescedRead(out_v[v_arg_row][ss](b));
        }
#endif
        coalescedWrite(out_v[v_arg_row][ss](b),res);
      });
      MComputeTime += usecond();
    }
    accelerator_barrier();
    MViewTime -= usecond();
    VECTOR_VIEW_CLOSE(Uc_v);
    VECTOR_VIEW_CLOSE(out_v);
    MViewTime += usecond();
    MTotalTime += usecond();
    //grid_message("Finished calling: DhopInternal_gpu\n");
  }

  void DhopInternal_cpu(Stencil& stencil, const PhysicalGaugeField& Uc, const VFermionField& in, VFermionField& out, FermionField& tmp) {
    DhopInternal_gpu(stencil, Uc, in, out, tmp);
  }

  void DhopDagInternal_gpu(Stencil& stencil, const PhysicalGaugeField& Uc, const VFermionField& in, VFermionField& out, FermionField& tmp) {
    MCalls++;
    MTotalTime -= usecond();
    MMiscTime -= usecond();
    const int Narg            = numArg(in, out);
    const int NvirtualFermion = numVirtual(in, out);
    const int NvirtualLink    = NvirtualFermion*NvirtualFermion;
    const int Nsimd           = SiteSpinor::Nsimd();
    const int Nsite           = in[0].Grid()->oSites();
    const int Npoint          = geom_.npoint;
    const int Npoint_hop      = geom_.npoint-1; // all but the self-coupling term

    assert(n_arg_ == Narg);

    SimpleCompressor<SiteSpinor> compressor;
    MMiscTime += usecond();

    copyToTmp5dField(in, tmp, Nsite, NvirtualFermion, Narg); // timed inside

    MCommTime-=usecond();
    stencil.HaloExchange(tmp, compressor);
    MCommTime+=usecond();

    MViewTime -= usecond();
    VECTOR_VIEW_OPEN(Uc, Uc_v, AcceleratorRead);
    autoView(tmp_v, tmp, AcceleratorRead);
    autoView(stencil_v, stencil, AcceleratorRead);
    auto tmp_p = &tmp_v[0];
    VECTOR_VIEW_OPEN(out, out_v, AcceleratorWrite);
    RealD* dag_factor_p = &dag_factor_[0];
    auto nbasis_global__ = nbasis_global_;
    MViewTime += usecond();

    for(int v_col=0; v_col<NvirtualFermion; v_col++) {
      typedef decltype(coalescedRead(tmp_v[0]))    calcVector;
      typedef decltype(coalescedRead(tmp_v[0](0))) calcComplex;

      MComputeTime -= usecond();
      accelerator_forNB(idx, Nsite*NvirtualFermion*Narg*NbasisVirtual, Nsimd, {
              int _idx  = idx;
        const int b     = _idx%NbasisVirtual; _idx/=NbasisVirtual;
        const int arg   = _idx%Narg; _idx/=Narg;
        const int v_row = _idx%NvirtualFermion; _idx/=NvirtualFermion;
        const int ss    = _idx%Nsite; _idx/=Nsite;

        const int v_arg_col = arg*NvirtualFermion+v_col;
        const int v_arg_row = arg*NvirtualFermion+v_row;
        const int v_row_col = v_row * NvirtualFermion + v_col;
        const int v_col_row = v_col * NvirtualFermion + v_row;
        const int sF        = ss*NvirtualFermion*Narg+v_col*Narg+arg; // needed for stencil access

        const int b_global = v_row*NvirtualFermion+b;

#if defined(ROW_MAJOR)
        const int v_link = v_row_col;
#else // =  COL_MAJOR
        const int v_link = v_col_row;
#endif

        calcComplex res;
        calcVector nbr;
        int ptype;
        StencilEntry *SE_MA;

#if defined(REFERENCE_SUMMATION_ORDER)
        res = Zero();
#else
        if (v_col == 0)
          res = Zero();
        else
          res = coalescedRead(out_v[v_arg_row][ss](b));
#endif

        for(int point=0; point<Npoint_hop; point++) {
          SE_MA=stencil_v.GetEntry(ptype,point,sF);

          if(SE_MA->_is_local) {
            nbr = coalescedReadPermute(tmp_v[SE_MA->_offset],ptype,SE_MA->_permute);
          } else {
            nbr = coalescedRead(stencil_v.CommBuf()[SE_MA->_offset]);
          }
          acceleratorSynchronise();

          for(int bb=0;bb<NbasisVirtual;bb++) {
            const int bb_global = v_row*NvirtualFermion+bb;
#if defined(TENSOR_LAYOUT)
            res = res + dag_factor_p[b_global*nbasis_global__+bb_global] * coalescedRead(Uc_v[v_link][ss](point)(b,bb))*nbr(bb);
#else
            res = res + dag_factor_p[b_global*nbasis_global__+bb_global] * coalescedRead(Uc_v[v_link*Npoint+point][ss](b,bb))*nbr(bb);
#endif
          }
        }
#if defined(REFERENCE_SUMMATION_ORDER)
        if (v_col != 0) {
          res = res + coalescedRead(out_v[v_arg_row][ss](b));
        }
#endif
        coalescedWrite(out_v[v_arg_row][ss](b),res);
      });
      MComputeTime += usecond();
    }
    accelerator_barrier();
    MViewTime -= usecond();
    VECTOR_VIEW_CLOSE(Uc_v);
    VECTOR_VIEW_CLOSE(out_v);
    MViewTime += usecond();
    MTotalTime += usecond();
    //grid_message("Finished calling: DhopDagInternal_gpu\n");
  }

  void DhopDagInternal_cpu(Stencil& stencil, const PhysicalGaugeField& Uc, const VFermionField& in, VFermionField& out, FermionField& tmp) {
    DhopDagInternal_gpu(stencil, Uc, in, out, tmp);
  }

  void DhopDirInternal_gpu(Stencil& stencil, const PhysicalGaugeField& Uc, const VFermionField& in, VFermionField& out, FermionField& tmp, int point) {
    MCalls++;
    MTotalTime -= usecond();
    MMiscTime -= usecond();
    const int Narg            = numArg(in, out);
    const int NvirtualFermion = numVirtual(in, out);
    const int NvirtualLink    = NvirtualFermion*NvirtualFermion;
    const int Nsimd           = SiteSpinor::Nsimd();
    const int Nsite           = in[0].Grid()->oSites();
    const int Npoint          = geom_.npoint;

    //grid_message("DhopDirInternal_gpu: n_arg_ = %d, Narg = %d\n", n_arg_, Narg);
    assert(n_arg_ == Narg);

    SimpleCompressor<SiteSpinor> compressor;
    MMiscTime += usecond();

    copyToTmp5dField(in, tmp, Nsite, NvirtualFermion, Narg); // timed inside

    MCommTime-=usecond();
    stencil.HaloExchange(tmp, compressor);
    MCommTime+=usecond();

    MViewTime -= usecond();
    VECTOR_VIEW_OPEN(Uc, Uc_v, AcceleratorRead);
    autoView(tmp_v, tmp, AcceleratorRead);
    autoView(stencil_v, stencil, AcceleratorRead);
    auto tmp_p = &tmp_v[0];
    VECTOR_VIEW_OPEN(out, out_v, AcceleratorWrite);
    MViewTime += usecond();

    for(int v_col=0; v_col<NvirtualFermion; v_col++) {
      typedef decltype(coalescedRead(tmp_v[0]))    calcVector;
      typedef decltype(coalescedRead(tmp_v[0](0))) calcComplex;

      MComputeTime -= usecond();
      accelerator_forNB(idx, Nsite*NvirtualFermion*Narg*NbasisVirtual, Nsimd, {
              int _idx  = idx;
        const int b     = _idx%NbasisVirtual; _idx/=NbasisVirtual;
        const int arg   = _idx%Narg; _idx/=Narg;
        const int v_row = _idx%NvirtualFermion; _idx/=NvirtualFermion;
        const int ss    = _idx%Nsite; _idx/=Nsite;

        const int v_arg_col = arg*NvirtualFermion+v_col;
        const int v_arg_row = arg*NvirtualFermion+v_row;
        const int v_row_col = v_row * NvirtualFermion + v_col;
        const int v_col_row = v_col * NvirtualFermion + v_row;
        const int sF        = ss*NvirtualFermion*Narg+v_col*Narg+arg; // needed for stencil access

#if defined(ROW_MAJOR)
        const int v_link = v_row_col;
#else // =  COL_MAJOR
        const int v_link = v_col_row;
#endif

        calcComplex res;
        calcVector nbr;
        int ptype;
        StencilEntry *SE_MA;

#if defined(REFERENCE_SUMMATION_ORDER)
        res = Zero();
#else
        if (v_col == 0)
          res = Zero();
        else
          res = coalescedRead(out_v[v_arg_row][ss](b));
#endif

        SE_MA=stencil_v.GetEntry(ptype,point,sF);

        if(SE_MA->_is_local) {
          nbr = coalescedReadPermute(tmp_v[SE_MA->_offset],ptype,SE_MA->_permute);
        } else {
          nbr = coalescedRead(stencil_v.CommBuf()[SE_MA->_offset]);
        }
        acceleratorSynchronise();

        for(int bb=0;bb<NbasisVirtual;bb++) {
#if defined(TENSOR_LAYOUT)
          res = res + coalescedRead(Uc_v[v_link][ss](point)(b,bb))*nbr(bb);
#else
          res = res + coalescedRead(Uc_v[v_link*Npoint+point][ss](b,bb))*nbr(bb);
#endif
        }
#if defined(REFERENCE_SUMMATION_ORDER)
        if (v_col != 0) {
          res = res + coalescedRead(out_v[v_arg_row][ss](b));
        }
#endif
        coalescedWrite(out_v[v_arg_row][ss](b),res);
      });
      MComputeTime += usecond();
    }
    accelerator_barrier();
    MViewTime -= usecond();
    VECTOR_VIEW_CLOSE(Uc_v);
    VECTOR_VIEW_CLOSE(out_v);
    MViewTime += usecond();
    MTotalTime += usecond();
    //grid_message("Finished calling: DhopDirInternal_gpu\n");
  }

  void DhopDirInternal_cpu(Stencil& stencil, const PhysicalGaugeField& Uc, const VFermionField& in, VFermionField& out, FermionField& tmp, int point) {
    DhopDirInternal_gpu(stencil, Uc, in, out, tmp, point);
  }

  void MooeeInternal_gpu(const PhysicalLinkField& UcSelf, const VFermionField& in, VFermionField& out) {
    MCalls++;
    MTotalTime -= usecond();
    MMiscTime -= usecond();
    const int Narg            = numArg(in, out);
    const int NvirtualFermion = numVirtual(in, out);
    const int NvirtualLink    = NvirtualFermion*NvirtualFermion;
    const int Nsimd           = SiteSpinor::Nsimd();
    const int Nsite           = in[0].Grid()->oSites();

    assert(n_arg_ == Narg);

    constantCheckerboard(in, out);

    MViewTime -= usecond();
    VECTOR_VIEW_OPEN(UcSelf, UcSelf_v, AcceleratorRead);
    VECTOR_VIEW_OPEN(in, in_v,   AcceleratorRead);
    VECTOR_VIEW_OPEN(out, out_v, AcceleratorWrite);
    MViewTime += usecond();

    for(int v_col=0; v_col<NvirtualFermion; v_col++) {
      typedef decltype(coalescedRead(in_v[0][0]))    calcVector;
      typedef decltype(coalescedRead(in_v[0][0](0))) calcComplex;

      MComputeTime -= usecond();
      accelerator_forNB(idx, Nsite*NvirtualFermion*Narg*NbasisVirtual, Nsimd, {
              int _idx  = idx;
        const int b     = _idx%NbasisVirtual; _idx/=NbasisVirtual;
        const int arg   = _idx%Narg; _idx/=Narg;
        const int v_row = _idx%NvirtualFermion; _idx/=NvirtualFermion;
        const int ss    = _idx%Nsite; _idx/=Nsite;

        const int v_arg_col = arg*NvirtualFermion+v_col;
        const int v_arg_row = arg*NvirtualFermion+v_row;
        const int v_row_col = v_row * NvirtualFermion + v_col;
        const int v_col_row = v_col * NvirtualFermion + v_row;
        const int sF        = ss*NvirtualFermion*Narg+v_col*Narg+arg; // needed for stencil access

#if defined(ROW_MAJOR)
        const int v_link = v_row_col;
#else // =  COL_MAJOR
        const int v_link = v_col_row;
#endif

        calcComplex res;
        calcVector nbr;
        int ptype;
        StencilEntry *SE_MA;

#if defined(REFERENCE_SUMMATION_ORDER)
        res = Zero();
#else
        if (v_col == 0)
          res = Zero();
        else
          res = coalescedRead(out_v[v_arg_row][ss](b));
#endif

        nbr = coalescedRead(in_v[v_arg_col][ss]);

        for(int bb=0;bb<NbasisVirtual;bb++) {
          res = res + coalescedRead(UcSelf_v[v_link][ss](b,bb))*nbr(bb);
        }
#if defined(REFERENCE_SUMMATION_ORDER)
        if (v_col != 0) {
          res = res + coalescedRead(out_v[v_arg_row][ss](b));
        }
#endif
        coalescedWrite(out_v[v_arg_row][ss](b),res);
      });
      MComputeTime += usecond();
    }
    accelerator_barrier();
    MViewTime -= usecond();
    VECTOR_VIEW_CLOSE(UcSelf_v);
    VECTOR_VIEW_CLOSE(in_v);
    VECTOR_VIEW_CLOSE(out_v);
    MViewTime += usecond();
    MTotalTime += usecond();
    //grid_message("Finished calling: MooeeInternal_gpu\n");
  }

  void MooeeInternal_cpu(const PhysicalLinkField& UcSelf, const VFermionField& in, VFermionField& out) {
    MooeeInternal_gpu(UcSelf, in, out);
  }

  void MooeeDagInternal_gpu(const PhysicalLinkField& UcSelf, const VFermionField& in, VFermionField& out) {
    MCalls++;
    MTotalTime -= usecond();
    MMiscTime -= usecond();
    const int Narg            = numArg(in, out);
    const int NvirtualFermion = numVirtual(in, out);
    const int NvirtualLink    = NvirtualFermion*NvirtualFermion;
    const int Nsimd           = SiteSpinor::Nsimd();
    const int Nsite           = in[0].Grid()->oSites();

    assert(n_arg_ == Narg);

    constantCheckerboard(in, out);

    MViewTime -= usecond();
    VECTOR_VIEW_OPEN(UcSelf, UcSelf_v, AcceleratorRead);
    VECTOR_VIEW_OPEN(in, in_v,   AcceleratorRead);
    VECTOR_VIEW_OPEN(out, out_v, AcceleratorWrite);
    RealD* dag_factor_p = &dag_factor_[0];
    auto nbasis_global__ = nbasis_global_;
    MViewTime += usecond();

    for(int v_col=0; v_col<NvirtualFermion; v_col++) {
      typedef decltype(coalescedRead(in_v[0][0]))    calcVector;
      typedef decltype(coalescedRead(in_v[0][0](0))) calcComplex;

      MComputeTime -= usecond();
      accelerator_forNB(idx, Nsite*NvirtualFermion*Narg*NbasisVirtual, Nsimd, {
              int _idx  = idx;
        const int b     = _idx%NbasisVirtual; _idx/=NbasisVirtual;
        const int arg   = _idx%Narg; _idx/=Narg;
        const int v_row = _idx%NvirtualFermion; _idx/=NvirtualFermion;
        const int ss    = _idx%Nsite; _idx/=Nsite;

        const int v_arg_col = arg*NvirtualFermion+v_col;
        const int v_arg_row = arg*NvirtualFermion+v_row;
        const int v_row_col = v_row * NvirtualFermion + v_col;
        const int v_col_row = v_col * NvirtualFermion + v_row;
        const int sF        = ss*NvirtualFermion*Narg+v_col*Narg+arg; // needed for stencil access

        const int b_global = v_row*NvirtualFermion+b;

#if defined(ROW_MAJOR)
        const int v_link = v_row_col;
#else // =  COL_MAJOR
        const int v_link = v_col_row;
#endif

        calcComplex res;
        calcVector nbr;
        int ptype;
        StencilEntry *SE_MA;

#if defined(REFERENCE_SUMMATION_ORDER)
        res = Zero();
#else
        if (v_col == 0)
          res = Zero();
        else
          res = coalescedRead(out_v[v_arg_row][ss](b));
#endif

        nbr = coalescedRead(in_v[v_arg_col][ss]);

        for(int bb=0;bb<NbasisVirtual;bb++) {
          const int bb_global = v_row*NvirtualFermion+bb;
          res = res + dag_factor_p[b_global*nbasis_global__+bb_global] * coalescedRead(UcSelf_v[v_link][ss](b,bb))*nbr(bb);
        }
#if defined(REFERENCE_SUMMATION_ORDER)
        if (v_col != 0) {
          res = res + coalescedRead(out_v[v_arg_row][ss](b));
        }
#endif
        coalescedWrite(out_v[v_arg_row][ss](b),res);
      });
      MComputeTime += usecond();
    }
    accelerator_barrier();
    MViewTime -= usecond();
    VECTOR_VIEW_CLOSE(UcSelf_v);
    VECTOR_VIEW_CLOSE(in_v);
    VECTOR_VIEW_CLOSE(out_v);
    MViewTime += usecond();
    MTotalTime += usecond();
    //grid_message("Finished calling: MooeeDagInternal_gpu\n");
  }

  void MooeeDagInternal_cpu(const PhysicalLinkField& UcSelf, const VFermionField& in, VFermionField& out) {
    MooeeDagInternal_gpu(UcSelf, in, out);
  }
};
