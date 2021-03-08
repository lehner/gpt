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

  virtual void M(const FermionField& in, uint64_t in_n_virtual, FermionField& out, uint64_t out_n_virtual) = 0;
  virtual void Mdag(const FermionField& in, uint64_t in_n_virtual, FermionField& out, uint64_t out_n_virtual) = 0;
  virtual void MdagM(const FermionField& in, uint64_t in_n_virtual, FermionField& out, uint64_t out_n_virtual) = 0;
  virtual void Mdiag(const FermionField& in, uint64_t in_n_virtual, FermionField& out, uint64_t out_n_virtual) { Mooee(in, in_n_virtual, out, out_n_virtual); }
  virtual void Mdir(const FermionField& in, uint64_t in_n_virtual, FermionField& out, uint64_t out_n_virtual, int dir, int disp) = 0;
  // virtual void MdirAll(const FermionField& in, uint64_t in_n_virtual, std::vector<FermionField>& out, uint64_t out_n_virtual) = 0; // TODO think about this one again!

  /////////////////////////////////////////////////////////////////////////////
  //                            half cb operations                           //
  /////////////////////////////////////////////////////////////////////////////

  virtual void Meooe(const FermionField& in, uint64_t in_n_virtual, FermionField& out, uint64_t out_n_virtual) = 0;
  virtual void MeooeDag(const FermionField& in, uint64_t in_n_virtual, FermionField& out, uint64_t out_n_virtual) = 0;
  virtual void Mooee(const FermionField& in, uint64_t in_n_virtual, FermionField& out, uint64_t out_n_virtual) = 0;
  virtual void MooeeDag(const FermionField& in, uint64_t in_n_virtual, FermionField& out, uint64_t out_n_virtual) = 0;
  virtual void MooeeInv(const FermionField& in, uint64_t in_n_virtual, FermionField& out, uint64_t out_n_virtual) = 0;
  virtual void MooeeInvDag(const FermionField& in, uint64_t in_n_virtual, FermionField& out, uint64_t out_n_virtual) = 0;

  /////////////////////////////////////////////////////////////////////////////
  //               non-hermitian hopping term; half cb or both               //
  /////////////////////////////////////////////////////////////////////////////

  virtual void Dhop(const FermionField& in, uint64_t in_n_virtual, FermionField& out, uint64_t out_n_virtual, int dag) = 0;
  virtual void DhopOE(const FermionField& in, uint64_t in_n_virtual, FermionField& out, uint64_t out_n_virtual, int dag) = 0;
  virtual void DhopEO(const FermionField& in, uint64_t in_n_virtual, FermionField& out, uint64_t out_n_virtual, int dag) = 0;
  virtual void DhopDir(const FermionField& in, uint64_t in_n_virtual, FermionField& out, uint64_t out_n_virtual, int dir, int disp) = 0;

  /////////////////////////////////////////////////////////////////////////////
  //                               dminus stuff                              //
  /////////////////////////////////////////////////////////////////////////////

  virtual void Dminus(const FermionField& in, uint64_t in_n_virtual, FermionField& out, uint64_t out_n_virtual) = 0;
  virtual void DminusDag(const FermionField& in, uint64_t in_n_virtual, FermionField& out, uint64_t out_n_virtual) = 0;

  /////////////////////////////////////////////////////////////////////////////
  //                              import/export                              //
  /////////////////////////////////////////////////////////////////////////////

  virtual void ImportPhysicalFermionSource(const FermionField& input, uint64_t input_n_virtual, FermionField& imported, uint64_t imported_n_virtual) = 0;
  virtual void ImportUnphysicalFermion(const FermionField& input, uint64_t input_n_virtual, FermionField& imported, uint64_t imported_n_virtual) = 0;
  virtual void ExportPhysicalFermionSolution(const FermionField& solution, uint64_t solution_n_virtual, FermionField& exported, uint64_t exported_n_virtual) = 0;
  virtual void ExportPhysicalFermionSource(const FermionField& solution, uint64_t solution_n_virtual, FermionField& exported, uint64_t exported_n_virtual) = 0;
};

template<class CComplex,int NbasisVirtual>
class MultiArgVirtualCoarsenedMatrix : public MultiArgFermionOperatorBase<PVector<Lattice<iVector<CComplex,NbasisVirtual>>>> {
public: // type definitions ///////////////////////////////////////////////////

  // site-wise types
  typedef         iVector<CComplex, NbasisVirtual>          SiteSpinor;
  typedef         iMatrix<CComplex, NbasisVirtual>          SiteMatrix;
  typedef iVector<iMatrix<CComplex, NbasisVirtual>, 2*Nd+1> DoubleStoredSiteMatrix;

  // lattice types = virtual fields
  typedef Lattice<SiteSpinor>             VirtualFermionField;
  typedef Lattice<SiteMatrix>             VirtualLinkField;
  typedef Lattice<SiteMatrix>             VirtualGridLayoutGaugeField;
  typedef Lattice<DoubleStoredSiteMatrix> VirtualDoubleStoredGaugeField;

  // choose gauge type depending on compilation
#if defined (TENSOR_LAYOUT)
  typedef VirtualDoubleStoredGaugeField  VirtualGaugeField;
#else
  typedef VirtualGridLayoutGaugeField    VirtualGaugeField;
#endif

  // physical fields, used internally
  typedef std::vector<VirtualFermionField> PhysicalFermionField;
  typedef std::vector<VirtualLinkField>    PhysicalLinkField;
  typedef std::vector<VirtualGaugeField>   PhysicalGaugeField;

  // used by the outside world
  typedef CartesianStencil<SiteSpinor,SiteSpinor,int> Stencil;
  typedef typename SiteSpinor::vector_type            vCoeff_t;
  typedef PVector<VirtualFermionField>                FermionField;
  typedef PVector<VirtualLinkField>                   LinkField;
  typedef PVector<VirtualGaugeField>                  GaugeField;

private: // member data ///////////////////////////////////////////////////////

  Geometry geom_;
  Geometry geomMultiArg_;

  GridBase* grid_;
  GridBase* cbGrid_;
  GridBase* gridMultiArg_;
  GridBase* cbGridMultiArg_;

  bool hermitianOverall_;
  bool hermitianSelf_;

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

  PhysicalFermionField tmp_;

  VirtualFermionField tmpMultiArg_;
  VirtualFermionField tmpEvenMultiArg_;
  VirtualFermionField tmpOddMultiArg_;

  Vector<RealD> dag_factor_;

  double MCalls;
  double MMiscTime;
  double MViewTime;
  double MView2Time;
  double MCopyTime;
  double MCommTime;
  double MComputeTime;
  double MTotalTime;

  // Vector<RealD> dagFactor_;

public: // member functions (implementing interface) //////////////////////////

  // grids
  GridBase* FermionGrid()         { return grid_; }
  GridBase* FermionRedBlackGrid() { return cbGrid_; }
  GridBase* GaugeGrid()           { return grid_; }
  GridBase* GaugeRedBlackGrid()   { return cbGrid_; }

  // info about diagonal term
  int ConstEE()     { return 0; }
  int isTrivialEE() { return 0; }

  // full cb operations
  void M(const FermionField& in, uint64_t in_n_virtual, FermionField& out, uint64_t out_n_virtual) override {
    MInternal(in, in_n_virtual, out, out_n_virtual);
#if 0 // only needed when we're separating self coupling from directions
    constantCheckerboard(in, out);
    Dhop(in, in_n_virtual, out, out_n_virtual, DaggerNo);
    // TODO move to external function START
    FermionField tmp_pvec;
    tmp_pvec.resize(tmp_.size());
    for(size_t i=0;i<tmp_.size();i++) {
      tmp_pvec(i) = &(tmp_[i]);
    }
    // TODO move to external function END
    Mooee(in, in_n_virtual, tmp_pvec, out_n_virtual);
    axpy(out, 1.0, out, tmp_pvec, out_n_virtual);
#endif
  }
  void Mdag(const FermionField& in, uint64_t in_n_virtual, FermionField& out, uint64_t out_n_virtual) override {
    MdagInternal(in, in_n_virtual, out, out_n_virtual);
  }
  void MdagM(const FermionField& in, uint64_t in_n_virtual, FermionField& out, uint64_t out_n_virtual) override {
    grid_printf_flush("TODO: implement this correctly\n");
    MInternal(in, in_n_virtual, out, out_n_virtual);
  }
  void Mdiag(const FermionField& in, uint64_t in_n_virtual, FermionField& out, uint64_t out_n_virtual) override {
    Mooee(in, in_n_virtual, out, out_n_virtual);
  }
  void Mdir(const FermionField& in, uint64_t in_n_virtual, FermionField& out, uint64_t out_n_virtual, int dir, int disp) override {
    DhopDir(in, in_n_virtual, out, out_n_virtual, dir, disp);
  }
  // void MdirAll(const FermionField& in, uint64_t in_n_virtual, std::vector<FermionField>& out, uint64_t out_n_virtual) override { // TODO think about this one again!
  //   DhopDirAll(in, in_n_virtual, out, out_n_virtual);
  // }

  // half cb operations
  void Meooe(const FermionField& in, uint64_t in_n_virtual, FermionField& out, uint64_t out_n_virtual) override {
    if(in[0].Checkerboard() == Odd) {
      DhopEO(in, in_n_virtual, out, out_n_virtual, DaggerNo);
    } else {
      DhopOE(in, in_n_virtual, out, out_n_virtual, DaggerNo);
    }
  }
  void MeooeDag(const FermionField& in, uint64_t in_n_virtual, FermionField& out, uint64_t out_n_virtual) override {
    if(in[0].Checkerboard() == Odd) {
      DhopEO(in, in_n_virtual, out, out_n_virtual, DaggerYes);
    } else {
      DhopOE(in, in_n_virtual, out, out_n_virtual, DaggerYes);
    }
  }
  void Mooee(const FermionField& in, uint64_t in_n_virtual, FermionField& out, uint64_t out_n_virtual) override {
    if(in[0].Grid()->_isCheckerBoarded) {
      if(in[0].Checkerboard() == Odd) {
        MooeeInternal(UcSelfOdd_, in, in_n_virtual, out, out_n_virtual, DaggerNo);
      } else {
        MooeeInternal(UcSelfEven_, in, in_n_virtual, out, out_n_virtual, DaggerNo);
      }
    } else {
      MooeeInternal(UcSelf_, in, in_n_virtual, out, out_n_virtual, DaggerNo);
    }
  }
  void MooeeDag(const FermionField& in, uint64_t in_n_virtual, FermionField& out, uint64_t out_n_virtual) override {
    if(in[0].Grid()->_isCheckerBoarded) {
      if(in[0].Checkerboard() == Odd) {
        MooeeInternal(UcSelfOdd_, in, in_n_virtual, out, out_n_virtual, DaggerYes);
      } else {
        MooeeInternal(UcSelfEven_, in, in_n_virtual, out, out_n_virtual, DaggerYes);
      }
    } else {
      MooeeInternal(UcSelf_, in, in_n_virtual, out, out_n_virtual, DaggerYes);
    }
  }
  void MooeeInv(const FermionField& in, uint64_t in_n_virtual, FermionField& out, uint64_t out_n_virtual) override {
    if(in[0].Grid()->_isCheckerBoarded) {
      if(in[0].Checkerboard() == Odd) {
        MooeeInternal(UcSelfInvOdd_, in, in_n_virtual, out, out_n_virtual, DaggerNo);
      } else {
        MooeeInternal(UcSelfInvEven_, in, in_n_virtual, out, out_n_virtual, DaggerNo);
      }
    } else {
      MooeeInternal(UcSelfInv_, in, in_n_virtual, out, out_n_virtual, DaggerNo);
    }
  }
  void MooeeInvDag(const FermionField& in, uint64_t in_n_virtual, FermionField& out, uint64_t out_n_virtual) override {
    if(in[0].Grid()->_isCheckerBoarded) {
      if(in[0].Checkerboard() == Odd) {
        MooeeInternal(UcSelfInvOdd_, in, in_n_virtual, out, out_n_virtual, DaggerYes);
      } else {
        MooeeInternal(UcSelfInvEven_, in, in_n_virtual, out, out_n_virtual, DaggerYes);
      }
    } else {
      MooeeInternal(UcSelfInv_, in, in_n_virtual, out, out_n_virtual, DaggerYes);
    }
  }

  // non-hermitian hopping term; half cb or both
  void Dhop(const FermionField& in, uint64_t in_n_virtual, FermionField& out, uint64_t out_n_virtual, int dag) override {
    conformable(grid_, in); // verifies full grid
    conformable(in[0].Grid(), out);

    constantCheckerboard(in, out);

    DhopInternal(stencilMultiArg_, Uc_, in, in_n_virtual, out, out_n_virtual, tmpMultiArg_, dag);
  }
  void DhopOE(const FermionField& in, uint64_t in_n_virtual, FermionField& out, uint64_t out_n_virtual, int dag) override {
    conformable(cbGrid_, in);       // verifies half grid
    conformable(in[0].Grid(), out); // drops the cb check

    assert(checkerBoardIs(in, Even));
    changingCheckerboard(in, out);

    DhopInternal(stencilEvenMultiArg_, UcOdd_, in, in_n_virtual, out, out_n_virtual, tmpEvenMultiArg_, dag);
  }
  void DhopEO(const FermionField& in, uint64_t in_n_virtual, FermionField& out, uint64_t out_n_virtual, int dag) override {
    conformable(cbGrid_, in);       // verifies half grid
    conformable(in[0].Grid(), out); // drops the cb check

    assert(checkerBoardIs(in, Odd));
    changingCheckerboard(in, out);

    DhopInternal(stencilOddMultiArg_, UcEven_, in, in_n_virtual, out, out_n_virtual, tmpOddMultiArg_, dag);
  }
  void DhopDir(const FermionField& in, uint64_t in_n_virtual, FermionField& out, uint64_t out_n_virtual, int dir, int disp) override {
    conformable(grid_, in); // verifies full grid
    conformable(in[0].Grid(), out);

    constantCheckerboard(in, out);

    DhopDirInternal(stencilMultiArg_, Uc_, in, in_n_virtual, out, out_n_virtual, tmpMultiArg_, geom_.point(dir, disp));
  }

  // dminus stuff
  void Dminus(const FermionField& in, uint64_t in_n_virtual, FermionField& out, uint64_t out_n_virtual) override {
    assert(0 && "TODO: implement this");
  }
  void DminusDag(const FermionField& in, uint64_t in_n_virtual, FermionField& out, uint64_t out_n_virtual) override {
    assert(0 && "TODO: implement this");
  }

  // import/export
  void ImportPhysicalFermionSource(const FermionField& input, uint64_t input_n_virtual, FermionField& imported, uint64_t imported_n_virtual) override {
    assert(0 && "TODO: implement this");
  }
  void ImportUnphysicalFermion(const FermionField& input, uint64_t input_n_virtual, FermionField& imported, uint64_t imported_n_virtual) override {
    assert(0 && "TODO: implement this");
  }
  void ExportPhysicalFermionSolution(const FermionField& solution, uint64_t solution_n_virtual, FermionField& exported, uint64_t exported_n_virtual) override {
    assert(0 && "TODO: implement this");
  }
  void ExportPhysicalFermionSource(const FermionField& solution, uint64_t solution_n_virtual, FermionField& exported, uint64_t exported_n_virtual) override {
    assert(0 && "TODO: implement this");
  }

public: // member functions (additional) //////////////////////////////////////

  // helpers
  void ImportGauge(const PVector<VirtualLinkField>& Uc, const PVector<VirtualLinkField>& UcSelfInv) {
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
    VECTOR_VIEW_OPEN_POINTER(Uc_, Uc_member_v, Uc_member_p, AcceleratorWrite);
    VECTOR_VIEW_OPEN_POINTER(Uc,  Uc_arg_v,    Uc_arg_p,    AcceleratorRead);
    for(int v=0; v<NvirtualLink; v++) {
      const int v_row = v%NvirtualFermion; const int v_col = v/NvirtualFermion; // NOTE: comes in from gpt with row faster index -> col-major order
#if defined(ROW_MAJOR)
      const int v_link = v_row * NvirtualFermion + v_col; // change to col faster index -> row-major order -> with    transpose
#else // =  COL_MAJOR
      const int v_link = v_col * NvirtualFermion + v_row; // keep      row faster index -> col-major order -> without transpose
#endif
      // new layout with Lorentz in tensor
      for(int p=0; p<Npoint; p++) {
        accelerator_for(ss, Nsite, Nsimd, {
          coalescedWrite(Uc_member_p[v_link][ss](p), coalescedRead(Uc_arg_p[p*NvirtualLink+v][ss]));
        });
      }
    }
    VECTOR_VIEW_CLOSE_POINTER(Uc_member_v, Uc_member_p);
    VECTOR_VIEW_CLOSE_POINTER(Uc_arg_v, Uc_arg_p);
#else // grid's layout with Lorentz in std::vector
    assert(Uc.size()        == Uc_.size());
    assert(UcSelfInv.size() == UcSelfInv_.size());

    conformable(GaugeGrid(), Uc);
    conformable(GaugeGrid(), UcSelfInv);

    VECTOR_VIEW_OPEN_POINTER(Uc_, Uc_member_v, Uc_member_p, AcceleratorWrite);
    VECTOR_VIEW_OPEN_POINTER(Uc,  Uc_arg_v,    Uc_arg_p,    AcceleratorRead);
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

        accelerator_for(ss, Nsite, Nsimd, {
          // coalescedWrite(Uc_member_p[v_point_link][ss], coalescedRead(Uc_arg_p[p*NvirtualLink+v][ss]));
          coalescedWrite(Uc_member_p[v_link_point][ss], coalescedRead(Uc_arg_p[p*NvirtualLink+v][ss]));
        });
      }
    }
    VECTOR_VIEW_CLOSE_POINTER(Uc_member_v, Uc_member_p);
    VECTOR_VIEW_CLOSE_POINTER(Uc_arg_v, Uc_arg_p);
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

    grid_printf_flush("ImportGauge of new Coarse Operator finished\n");
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

    grid_printf_flush("VirtualCoarsenedMatrix::PickCheckerboards finished\n");
  }

  void Report(int Nvec) {
    assert(Nvec == n_arg_);
    RealD Nproc = grid_->_Nprocessors;
    RealD Nnode = grid_->NodeCount();
    RealD volume = 1;
    Coordinate latt = grid_->GlobalDimensions();
    for(int mu=0;mu<Nd;mu++) volume=volume*latt[mu];
    RealD nbasis = NbasisVirtual * fermion_n_virtual_;

    if ( MCalls > 0 ) {
      grid_printf_flush("#### M calls report\n");
      grid_printf_flush("CoarseOperator Number of Calls                         : %d\n", (int)MCalls);
      grid_printf_flush("CoarseOperator MiscTime   /Calls, MiscTime    : %10.2f us, %10.2f us (= %6.2f %%)\n", MMiscTime   /MCalls, MMiscTime,    MMiscTime   /MTotalTime*100);
      grid_printf_flush("CoarseOperator ViewTime   /Calls, ViewTime    : %10.2f us, %10.2f us (= %6.2f %%)\n", MViewTime   /MCalls, MViewTime,    MViewTime   /MTotalTime*100);
      grid_printf_flush("CoarseOperator View2Time  /Calls, View2Time   : %10.2f us, %10.2f us (= %6.2f %%)\n", MView2Time  /MCalls, MView2Time,   MView2Time  /MTotalTime*100);
      grid_printf_flush("CoarseOperator CopyTime   /Calls, CopyTime    : %10.2f us, %10.2f us (= %6.2f %%)\n", MCopyTime   /MCalls, MCopyTime,    MCopyTime   /MTotalTime*100);
      grid_printf_flush("CoarseOperator CommTime   /Calls, CommTime    : %10.2f us, %10.2f us (= %6.2f %%)\n", MCommTime   /MCalls, MCommTime,    MCommTime   /MTotalTime*100);
      grid_printf_flush("CoarseOperator ComputeTime/Calls, ComputeTime : %10.2f us, %10.2f us (= %6.2f %%)\n", MComputeTime/MCalls, MComputeTime, MComputeTime/MTotalTime*100);
      grid_printf_flush("CoarseOperator TotalTime  /Calls, TotalTime   : %10.2f us, %10.2f us (= %6.2f %%)\n", MTotalTime  /MCalls, MTotalTime,   MTotalTime  /MTotalTime*100);

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
      grid_printf_flush("CoarseOperator Average mflops/s, mbytes/s per call                : %.0f, %.0f\n", mflops, mbytes);
      grid_printf_flush("CoarseOperator Average mflops/s, mbytes/s per call per rank       : %.0f, %.0f\n", mflops/Nproc, mbytes/Nproc);
      grid_printf_flush("CoarseOperator Average mflops/s, mbytes/s per call per node       : %.0f, %.0f\n", mflops/Nnode, mbytes/Nnode);

      RealD Fullmflops = flop_per_site*volume*MCalls/(MTotalTime);
      RealD Fullmbytes = byte_per_site*volume*MCalls/(MTotalTime);
      grid_printf_flush("CoarseOperator Average mflops/s, mbytes/s per call (full)         : %.0f, %.0f\n", Fullmflops, Fullmbytes);
      grid_printf_flush("CoarseOperator Average mflops/s, mbytes/s per call per rank (full): %.0f, %.0f\n", Fullmflops/Nproc, Fullmbytes/Nproc);
      grid_printf_flush("CoarseOperator Average mflops/s, mbytes/s per call per node (full): %.0f, %.0f\n", Fullmflops/Nnode, Fullmbytes/Nnode);

      // grid_printf_flush("CoarseOperator Stencil\n"); stencil_.Report();
      // grid_printf_flush("CoarseOperator StencilEven\n"); stencilEven_.Report();
      // grid_printf_flush("CoarseOperator StencilOdd\n"); stencilOdd_.Report();
      grid_printf_flush("CoarseOperator StencilMultiArg\n"); stencilMultiArg_.Report();
      // grid_printf_flush("CoarseOperator StencilMultiArgEven\n"); stencilMultiArgEven_.Report();
      // grid_printf_flush("CoarseOperator StencilMultiArgOdd\n"); stencilMultiArgOdd_.Report();
    }
    grid_printf_flush("Report of new Coarse Operator finished\n");
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

    // stencil_.ZeroCounters();
    // stencilEven_.ZeroCounters();
    // stencilOdd_.ZeroCounters();
    stencilMultiArg_.ZeroCounters();
    // stencilMultiArgEven_.ZeroCounters();
    // stencilMultiArgOdd_.ZeroCounters();
  }

  void copyToTmp5dField(const FermionField& in, VirtualFermionField& tmp, int Nsite, int NvirtualFermion, int Narg) {
    MView2Time-=usecond();
    VECTOR_VIEW_OPEN_POINTER(in, in_v, in_p, AcceleratorRead);
    autoView(tmp_v, tmp, AcceleratorWrite);
    MView2Time+=usecond();
    accelerator_for(sF, Nsite*NvirtualFermion*Narg, SiteSpinor::Nsimd(), {
        int _sF   = sF;                  // this does fastest to slowest from top to bottom
        const int arg   = _sF%Narg;            _sF/=Narg;
        const int v_col = _sF%NvirtualFermion; _sF/=NvirtualFermion;
        const int sU    = _sF%Nsite;           _sF/=Nsite;
        coalescedWrite(tmp_v[sF], in_v[arg*NvirtualFermion+v_col](sU));
        // grid_printf_flush("COPY: sF = %4d, arg = %4d, sU = %4d, v_col = %4d\n", sF, arg, sU, v_col); fflush(stdout);
      });
    MCopyTime+=usecond();
    MView2Time-=usecond();
    VECTOR_VIEW_CLOSE_POINTER(in_v, in_p);
    MView2Time+=usecond();
  }

  // constructors
  MultiArgVirtualCoarsenedMatrix(const PVector<VirtualLinkField>& Uc,
                                 const PVector<VirtualLinkField>& UcSelfInv,
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
    , hermitianSelf_(false)
    , link_n_virtual_(UcSelfInv.size())
    , fermion_n_virtual_(uint64_t(sqrt(UcSelfInv.size())))
    , n_arg_(numArg)
    , nbasis_global_(fermion_n_virtual_*NbasisVirtual)
    // , stencil_(grid_, geom_.npoint, Even, geom_.directions, geom_.displacements, 0)
    // , stencilEven_(cbGrid_, geom_.npoint, Even, geom_.directions, geom_.displacements, 0)
    // , stencilOdd_(cbGrid_, geom_.npoint, Odd, geom_.directions, geom_.displacements, 0)
    , stencilMultiArg_(gridMultiArg_, geomMultiArg_.npoint, Even, geomMultiArg_.directions, geomMultiArg_.displacements, 0)
    , stencilEvenMultiArg_(cbGridMultiArg_, geomMultiArg_.npoint, Even, geomMultiArg_.directions, geomMultiArg_.displacements, 0)
    , stencilOddMultiArg_(cbGridMultiArg_, geomMultiArg_.npoint, Odd, geomMultiArg_.directions, geomMultiArg_.displacements, 0)
#if 0 // for separating the self-stencil from the directions
    , stencilMultiArg_(gridMultiArg_, geomMultiArg_.npoint-1, Even, geomMultiArg_.directions, geomMultiArg_.displacements, 0)
    , stencilEvenMultiArg_(cbGridMultiArg_, geomMultiArg_.npoint-1, Even, geomMultiArg_.directions, geomMultiArg_.displacements, 0)
    , stencilOddMultiArg_(cbGridMultiArg_, geomMultiArg_.npoint-1, Odd, geomMultiArg_.directions, geomMultiArg_.displacements, 0)
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
    , tmp_(fermion_n_virtual_*n_arg_, grid_) // needed for temporary in M
    , tmpMultiArg_(gridMultiArg_)
    , tmpEvenMultiArg_(cbGridMultiArg_)
    , tmpOddMultiArg_(cbGridMultiArg_)
    , dag_factor_(nbasis_global_*nbasis_global_)
  {
    grid_->show_decomposition();
    cbGrid_->show_decomposition();
    gridMultiArg_->show_decomposition();
    cbGridMultiArg_->show_decomposition();
    ImportGauge(Uc, UcSelfInv);
    PickCheckerboards();
    ZeroCounters();

    // need to set these here explicitly since I can't pick a cb for them!
    tmpEvenMultiArg_.Checkerboard() = Even;
    tmpOddMultiArg_.Checkerboard() = Odd;

    fillFactor();

    assert(n_arg_ == 1); // Limit to 1 for now!

    reportVersion();
  }

private: // member functions //////////////////////////////////////////////////

  int numArg(const FermionField& a, uint64_t a_n_virtual, const FermionField& b, uint64_t b_n_virtual) const {
    int a_size = a.size();
    int b_size = b.size();
    assert(a_size == b_size);
    assert(a_n_virtual == b_n_virtual);
    assert(a_n_virtual == fermion_n_virtual_);
    assert(a_size >= a_n_virtual);
    assert(a_size % a_n_virtual == 0);
    return a_size / a_n_virtual;
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

  void reportVersion() {
#if defined(ROW_MAJOR)
    grid_printf_flush("Creating coarse operator link matrices in layout: row    major\n");
#else
    grid_printf_flush("Creating coarse operator link matrices in layout: column major\n");
#endif

#if defined(TENSOR_LAYOUT)
    grid_printf_flush("Creating coarse operator link matrices using: grid tensors\n");
#else
    grid_printf_flush("Creating coarse operator link matrices using: std::vector (as in grid)\n");
#endif

#if defined(REFERENCE_SUMMATION_ORDER)
    grid_printf_flush("Creating coarse operator with summation order: reference\n");
#else
    grid_printf_flush("Creating coarse operator with summation order: modified\n");
#endif
  }

public: // kernel functions TODO: move somewhere else ////////////////////////

#if 0 // only needed for separating self-stencil from directions
  template<class sobj>
  void axpy(FermionField& ret, sobj a, const FermionField& x, const FermionField& y, uint64_t n_virtual) const {
    const int Narg     = numArg(ret, n_virtual, x, n_virtual);
    const int Nsite    = x[0].Grid()->oSites();
    const int Nsimd    = SiteSpinor::Nsimd();
    const int Nlattice = ret.size();

    assert(ret.size() == x.size());
    assert(y.size()   == y.size());
    assert(Narg * fermion_n_virtual_ == Nlattice);

    constantCheckerboard(x, ret);
    conformable(ret, x);
    conformable(x, y);
    VECTOR_VIEW_OPEN_POINTER(ret, ret_v, ret_p, AcceleratorWrite);
    VECTOR_VIEW_OPEN_POINTER(x,   x_v,   x_p,   AcceleratorRead);
    VECTOR_VIEW_OPEN_POINTER(y,   y_v,   y_p,   AcceleratorRead);
    accelerator_for(idx, Nlattice*Nsite, Nsimd, {
            int _idx = idx;
      const int ss   = _idx%Nsite;    _idx/=Nsite;
      const int l    = _idx%Nlattice; _idx/=Nlattice;
      auto tmp = a * coalescedRead(x_p[l][ss]) + coalescedRead(y_p[l][ss]);
      coalescedWrite(ret_p[l][ss], tmp);
    });
    VECTOR_VIEW_CLOSE_POINTER(ret_v, ret_p);
    VECTOR_VIEW_CLOSE_POINTER(x_v,   x_p);
    VECTOR_VIEW_CLOSE_POINTER(y_v,   y_p);
  }
#endif

  void MInternal(const FermionField& in, uint64_t in_n_virtual, FermionField& out, uint64_t out_n_virtual) {
#if defined(GRID_CUDA) || defined(GRID_HIP)
    MInternal_gpu(in, in_n_virtual, out, out_n_virtual);
#else
    MInternal_cpu(in, in_n_virtual, out, out_n_virtual);
#endif
  }

  void MdagInternal(const FermionField& in, uint64_t in_n_virtual, FermionField& out, uint64_t out_n_virtual) {
#if defined(GRID_CUDA) || defined(GRID_HIP)
    MdagInternal_gpu(in, in_n_virtual, out, out_n_virtual);
#else
    MdagInternal_cpu(in, in_n_virtual, out, out_n_virtual);
#endif
  }

  void DhopInternal(Stencil& stencil, const PhysicalGaugeField& Uc, const FermionField& in, uint64_t in_n_virtual, FermionField& out, uint64_t out_n_virtual, VirtualFermionField& tmp, int dag) {
#if defined(GRID_CUDA) || defined(GRID_HIP)
    if(dag == DaggerYes)
      DhopDagInternal_gpu(stencil, Uc, in, in_n_virtual, out, out_n_virtual, tmp);
    else
      DhopInternal_gpu(stencil, Uc, in, in_n_virtual, out, out_n_virtual, tmp);
#else
    if(dag == DaggerYes)
      DhopDagInternal_cpu(stencil, Uc, in, in_n_virtual, out, out_n_virtual, tmp);
    else
      DhopInternal_cpu(stencil, Uc, in, in_n_virtual, out, out_n_virtual, tmp);
#endif
  }

  void DhopDirInternal(Stencil& stencil, const PhysicalGaugeField& Uc, const FermionField& in, uint64_t in_n_virtual, FermionField& out, uint64_t out_n_virtual, VirtualFermionField& tmp, int point) {
#if defined(GRID_CUDA) || defined(GRID_HIP)
    DhopDirInternal_gpu(stencil, Uc, in, in_n_virtual, out, out_n_virtual, tmp, point);
#else
    DhopDirInternal_cpu(stencil, Uc, in, in_n_virtual, out, out_n_virtual, tmp, point);
#endif
  }

  void MooeeInternal(const PhysicalLinkField& UcSelf, const FermionField& in, uint64_t in_n_virtual, FermionField& out, uint64_t out_n_virtual, int dag) {
#if defined(GRID_CUDA) || defined(GRID_HIP)
    if(dag == DaggerYes)
      MooeeDagInternal_gpu(UcSelf, in, in_n_virtual, out, out_n_virtual);
    else
      MooeeInternal_gpu(UcSelf, in, in_n_virtual, out, out_n_virtual);
#else
    if(dag == DaggerYes)
      MooeeDagInternal_cpu(UcSelf, in, in_n_virtual, out, out_n_virtual);
    else
      MooeeInternal_cpu(UcSelf, in, in_n_virtual, out, out_n_virtual);
#endif
  }

  void MInternal_gpu(const FermionField& in, uint64_t in_n_virtual, FermionField& out, uint64_t out_n_virtual) {
    // NOTE: corresponds to "M_finegrained_loopinternal_tensorlayout_parchange_commsreduce" in test file
    // NOTE: version with additional parallelism over output virtual index + reducing comms by temporary 5d object -- Lorentz in tensor
    MCalls++;
    MTotalTime -= usecond();
    MMiscTime -= usecond();
    const int Narg            = numArg(in, in_n_virtual, out, out_n_virtual);
    const int NvirtualFermion = in_n_virtual;
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
    VECTOR_VIEW_OPEN_POINTER(Uc_, Uc_v, Uc_p, AcceleratorRead);
    autoView(tmpMultiArg_v, tmpMultiArg_, AcceleratorRead);
    autoView(stencilMultiArg_v, stencilMultiArg_, AcceleratorRead);
    VECTOR_VIEW_OPEN_POINTER(out, out_v, out_p, AcceleratorWrite);
    MViewTime += usecond();

    for(int v_col=0; v_col<NvirtualFermion; v_col++) {
      typedef decltype(coalescedRead(tmpMultiArg_v[0]))    calcVector;
      typedef decltype(coalescedRead(tmpMultiArg_v[0](0))) calcComplex;

      MComputeTime -= usecond();
      accelerator_for(idx, Nsite*NvirtualFermion*Narg*NbasisVirtual, Nsimd, {
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
          res = coalescedRead(out_p[v_arg_row][ss](b));
#endif

        for(int point=0; point<Npoint; point++) {
          SE_MA=stencilMultiArg_v.GetEntry(ptype,point,sF);

          if(SE_MA->_is_local) {
            nbr = coalescedReadPermute(tmpMultiArg_v[SE_MA->_offset],ptype,SE_MA->_permute);
          } else {
            nbr = coalescedRead(stencilMultiArg_v.CommBuf()[SE_MA->_offset]);
          }
          acceleratorSynchronise();

          for(int bb=0;bb<NbasisVirtual;bb++) {
#if defined(TENSOR_LAYOUT)
            res = res + coalescedRead(Uc_p[v_link][ss](point)(b,bb))*nbr(bb);
#else
            res = res + coalescedRead(Uc_p[v_link*Npoint+point][ss](b,bb))*nbr(bb);
#endif
          }
        }
#if defined(REFERENCE_SUMMATION_ORDER)
        if (v_col != 0) {
          res = res + coalescedRead(out_p[v_arg_row][ss](b));
        }
#endif
        coalescedWrite(out_p[v_arg_row][ss](b),res);
      });
      MComputeTime += usecond();
    }
    MViewTime -= usecond();
    VECTOR_VIEW_CLOSE_POINTER(Uc_v, Uc_p);
    VECTOR_VIEW_CLOSE_POINTER(out_v, out_p);
    MViewTime += usecond();
    MTotalTime += usecond();
    grid_printf_flush("Finished calling: MInternal_gpu\n");
  }

  void MInternal_cpu(const FermionField& in, uint64_t in_n_virtual, FermionField& out, uint64_t out_n_virtual) {
    // NOTE: corresponds to "M_loopinternal_tensorlayout_parchange_commsreduce" in test file
    // NOTE: version with additional parallelism over output virtual index + reducing comms by temporary 5d object -- Lorentz in tensor
    MCalls++;
    MTotalTime -= usecond();
    MMiscTime -= usecond();
    const int Narg            = numArg(in, in_n_virtual, out, out_n_virtual);
    const int NvirtualFermion = in_n_virtual;
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
    VECTOR_VIEW_OPEN_POINTER(Uc_, Uc_v, Uc_p, AcceleratorRead);
    autoView(tmpMultiArg_v, tmpMultiArg_, AcceleratorRead);
    autoView(stencilMultiArg_v, stencilMultiArg_, AcceleratorRead);
    VECTOR_VIEW_OPEN_POINTER(out, out_v, out_p, AcceleratorWrite);
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
          res = coalescedRead(out_p[v_arg_row][ss]);
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
          res = res + coalescedRead(Uc_p[v_link][ss](point))*nbr;
#else
          res = res + coalescedRead(Uc_p[v_link*Npoint+point][ss])*nbr;
#endif
        }

#if defined(REFERENCE_SUMMATION_ORDER)
        if (v_col != 0) {
          res = res + coalescedRead(out_p[v_arg_row][ss]);
        }
#endif
        coalescedWrite(out_p[v_arg_row][ss],res);
      });
      MComputeTime += usecond();
    }
    MViewTime -= usecond();
    VECTOR_VIEW_CLOSE_POINTER(Uc_v, Uc_p);
    VECTOR_VIEW_CLOSE_POINTER(out_v, out_p);
    MViewTime += usecond();
    MTotalTime += usecond();
    grid_printf_flush("Finished calling: MInternal_cpu\n");
  }

  void MdagInternal_gpu(const FermionField& in, uint64_t in_n_virtual, FermionField& out, uint64_t out_n_virtual) {
    // NOTE: corresponds to "M_finegrained_loopinternal_tensorlayout_parchange_commsreduce" in test file
    // NOTE: version with additional parallelism over output virtual index + reducing comms by temporary 5d object -- Lorentz in tensor
    MCalls++;
    MTotalTime -= usecond();
    MMiscTime -= usecond();
    const int Narg            = numArg(in, in_n_virtual, out, out_n_virtual);
    const int NvirtualFermion = in_n_virtual;
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
    VECTOR_VIEW_OPEN_POINTER(Uc_, Uc_v, Uc_p, AcceleratorRead);
    autoView(tmpMultiArg_v, tmpMultiArg_, AcceleratorRead);
    autoView(stencilMultiArg_v, stencilMultiArg_, AcceleratorRead);
    VECTOR_VIEW_OPEN_POINTER(out, out_v, out_p, AcceleratorWrite);
    RealD* dag_factor_p = &dag_factor_[0];
    MViewTime += usecond();

    for(int v_col=0; v_col<NvirtualFermion; v_col++) {
      typedef decltype(coalescedRead(tmpMultiArg_v[0]))    calcVector;
      typedef decltype(coalescedRead(tmpMultiArg_v[0](0))) calcComplex;

      MComputeTime -= usecond();
      accelerator_for(idx, Nsite*NvirtualFermion*Narg*NbasisVirtual, Nsimd, {
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
          res = coalescedRead(out_p[v_arg_row][ss](b));
#endif

        for(int point=0; point<Npoint; point++) {
          SE_MA=stencilMultiArg_v.GetEntry(ptype,point,sF);

          if(SE_MA->_is_local) {
            nbr = coalescedReadPermute(tmpMultiArg_v[SE_MA->_offset],ptype,SE_MA->_permute);
          } else {
            nbr = coalescedRead(stencilMultiArg_v.CommBuf()[SE_MA->_offset]);
          }
          acceleratorSynchronise();

          for(int bb=0;bb<NbasisVirtual;bb++) {
            const int bb_global = v_row*NvirtualFermion+bb;
#if defined(TENSOR_LAYOUT)
            res = res + dag_factor_p[b_global*nbasis_global_+bb_global] * coalescedRead(Uc_p[v_link][ss](point)(b,bb))*nbr(bb);
#else
            res = res + dag_factor_p[b_global*nbasis_global_+bb_global] * coalescedRead(Uc_p[v_link*Npoint+point][ss](b,bb))*nbr(bb);
#endif
          }
        }
#if defined(REFERENCE_SUMMATION_ORDER)
        if (v_col != 0) {
          res = res + coalescedRead(out_p[v_arg_row][ss](b));
        }
#endif
        coalescedWrite(out_p[v_arg_row][ss](b),res);
      });
      MComputeTime += usecond();
    }
    MViewTime -= usecond();
    VECTOR_VIEW_CLOSE_POINTER(Uc_v, Uc_p);
    VECTOR_VIEW_CLOSE_POINTER(out_v, out_p);
    MViewTime += usecond();
    MTotalTime += usecond();
    grid_printf_flush("Finished calling: MInternal_gpu\n");
  }

  void MdagInternal_cpu(const FermionField& in, uint64_t in_n_virtual, FermionField& out, uint64_t out_n_virtual) {
    MdagInternal_gpu(in, in_n_virtual, out, out_n_virtual);
  }

  void DhopInternal_gpu(Stencil& stencil, const PhysicalGaugeField& Uc, const FermionField& in, uint64_t in_n_virtual, FermionField& out, uint64_t out_n_virtual, VirtualFermionField& tmp) {
    // NOTE: corresponds to "M_finegrained_loopinternal_tensorlayout_parchange_commsreduce" in test file
    // NOTE: version with additional parallelism over output virtual index + reducing comms by temporary 5d object -- Lorentz in tensor
    MCalls++;
    MTotalTime -= usecond();
    MMiscTime -= usecond();
    const int Narg            = numArg(in, in_n_virtual, out, out_n_virtual);
    const int NvirtualFermion = in_n_virtual;
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
    VECTOR_VIEW_OPEN_POINTER(Uc, Uc_v, Uc_p, AcceleratorRead);
    autoView(tmp_v, tmp, AcceleratorRead);
    autoView(stencil_v, stencil, AcceleratorRead);
    VECTOR_VIEW_OPEN_POINTER(out, out_v, out_p, AcceleratorWrite);
    MViewTime += usecond();

    for(int v_col=0; v_col<NvirtualFermion; v_col++) {
      typedef decltype(coalescedRead(tmp_v[0]))    calcVector;
      typedef decltype(coalescedRead(tmp_v[0](0))) calcComplex;

      MComputeTime -= usecond();
      accelerator_for(idx, Nsite*NvirtualFermion*Narg*NbasisVirtual, Nsimd, {
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
          res = coalescedRead(out_p[v_arg_row][ss](b));
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
#if defined(TENSOR_LAYOUT)
            res = res + coalescedRead(Uc_p[v_link][ss](point)(b,bb))*nbr(bb);
#else
            res = res + coalescedRead(Uc_p[v_link*Npoint+point][ss](b,bb))*nbr(bb);
#endif
          }
        }
#if defined(REFERENCE_SUMMATION_ORDER)
        if (v_col != 0) {
          res = res + coalescedRead(out_p[v_arg_row][ss](b));
        }
#endif
        coalescedWrite(out_p[v_arg_row][ss](b),res);
      });
      MComputeTime += usecond();
    }
    MViewTime -= usecond();
    VECTOR_VIEW_CLOSE_POINTER(Uc_v, Uc_p);
    VECTOR_VIEW_CLOSE_POINTER(out_v, out_p);
    MViewTime += usecond();
    MTotalTime += usecond();
    grid_printf_flush("Finished calling: DhopInternal_gpu\n");
  }

  void DhopInternal_cpu(Stencil& stencil, const PhysicalGaugeField& Uc, const FermionField& in, uint64_t in_n_virtual, FermionField& out, uint64_t out_n_virtual, VirtualFermionField& tmp) {
    DhopInternal_gpu(stencil, Uc, in, in_n_virtual, out, out_n_virtual, tmp);
  }

  void DhopDagInternal_gpu(Stencil& stencil, const PhysicalGaugeField& Uc, const FermionField& in, uint64_t in_n_virtual, FermionField& out, uint64_t out_n_virtual, VirtualFermionField& tmp) {
    // NOTE: corresponds to "M_finegrained_loopinternal_tensorlayout_parchange_commsreduce" in test file
    // NOTE: version with additional parallelism over output virtual index + reducing comms by temporary 5d object -- Lorentz in tensor
    MCalls++;
    MTotalTime -= usecond();
    MMiscTime -= usecond();
    const int Narg            = numArg(in, in_n_virtual, out, out_n_virtual);
    const int NvirtualFermion = in_n_virtual;
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
    VECTOR_VIEW_OPEN_POINTER(Uc, Uc_v, Uc_p, AcceleratorRead);
    autoView(tmp_v, tmp, AcceleratorRead);
    autoView(stencil_v, stencil, AcceleratorRead);
    VECTOR_VIEW_OPEN_POINTER(out, out_v, out_p, AcceleratorWrite);
    RealD* dag_factor_p = &dag_factor_[0];
    MViewTime += usecond();

    for(int v_col=0; v_col<NvirtualFermion; v_col++) {
      typedef decltype(coalescedRead(tmp_v[0]))    calcVector;
      typedef decltype(coalescedRead(tmp_v[0](0))) calcComplex;

      MComputeTime -= usecond();
      accelerator_for(idx, Nsite*NvirtualFermion*Narg*NbasisVirtual, Nsimd, {
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
          res = coalescedRead(out_p[v_arg_row][ss](b));
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
            res = res + dag_factor_p[b_global*nbasis_global_+bb_global] * coalescedRead(Uc_p[v_link][ss](point)(b,bb))*nbr(bb);
#else
            res = res + dag_factor_p[b_global*nbasis_global_+bb_global] * coalescedRead(Uc_p[v_link*Npoint+point][ss](b,bb))*nbr(bb);
#endif
          }
        }
#if defined(REFERENCE_SUMMATION_ORDER)
        if (v_col != 0) {
          res = res + coalescedRead(out_p[v_arg_row][ss](b));
        }
#endif
        coalescedWrite(out_p[v_arg_row][ss](b),res);
      });
      MComputeTime += usecond();
    }
    MViewTime -= usecond();
    VECTOR_VIEW_CLOSE_POINTER(Uc_v, Uc_p);
    VECTOR_VIEW_CLOSE_POINTER(out_v, out_p);
    MViewTime += usecond();
    MTotalTime += usecond();
    grid_printf_flush("Finished calling: DhopInternal_gpu\n");
  }

  void DhopDagInternal_cpu(Stencil& stencil, const PhysicalGaugeField& Uc, const FermionField& in, uint64_t in_n_virtual, FermionField& out, uint64_t out_n_virtual, VirtualFermionField& tmp) {
    DhopDagInternal_gpu(stencil, Uc, in, in_n_virtual, out, out_n_virtual, tmp);
  }

  void DhopDirInternal_gpu(Stencil& stencil, const PhysicalGaugeField& Uc, const FermionField& in, uint64_t in_n_virtual, FermionField& out, uint64_t out_n_virtual, VirtualFermionField& tmp, int point) {
    // NOTE: corresponds to "M_finegrained_loopinternal_tensorlayout_parchange_commsreduce" in test file
    // NOTE: version with additional parallelism over output virtual index + reducing comms by temporary 5d object -- Lorentz in tensor
    MCalls++;
    MTotalTime -= usecond();
    MMiscTime -= usecond();
    const int Narg            = numArg(in, in_n_virtual, out, out_n_virtual);
    const int NvirtualFermion = in_n_virtual;
    const int NvirtualLink    = NvirtualFermion*NvirtualFermion;
    const int Nsimd           = SiteSpinor::Nsimd();
    const int Nsite           = in[0].Grid()->oSites();
    const int Npoint          = geom_.npoint;

    assert(n_arg_ == Narg);

    SimpleCompressor<SiteSpinor> compressor;
    MMiscTime += usecond();

    copyToTmp5dField(in, tmp, Nsite, NvirtualFermion, Narg); // timed inside

    MCommTime-=usecond();
    stencil.HaloExchange(tmp, compressor);
    MCommTime+=usecond();

    MViewTime -= usecond();
    VECTOR_VIEW_OPEN_POINTER(Uc, Uc_v, Uc_p, AcceleratorRead);
    autoView(tmp_v, tmp, AcceleratorRead);
    autoView(stencil_v, stencil, AcceleratorRead);
    VECTOR_VIEW_OPEN_POINTER(out, out_v, out_p, AcceleratorWrite);
    MViewTime += usecond();

    for(int v_col=0; v_col<NvirtualFermion; v_col++) {
      typedef decltype(coalescedRead(tmp_v[0]))    calcVector;
      typedef decltype(coalescedRead(tmp_v[0](0))) calcComplex;

      MComputeTime -= usecond();
      accelerator_for(idx, Nsite*NvirtualFermion*Narg*NbasisVirtual, Nsimd, {
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
          res = coalescedRead(out_p[v_arg_row][ss](b));
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
          res = res + coalescedRead(Uc_p[v_link][ss](point)(b,bb))*nbr(bb);
#else
          res = res + coalescedRead(Uc_p[v_link*Npoint+point][ss](b,bb))*nbr(bb);
#endif
        }
#if defined(REFERENCE_SUMMATION_ORDER)
        if (v_col != 0) {
          res = res + coalescedRead(out_p[v_arg_row][ss](b));
        }
#endif
        coalescedWrite(out_p[v_arg_row][ss](b),res);
      });
      MComputeTime += usecond();
    }
    MViewTime -= usecond();
    VECTOR_VIEW_CLOSE_POINTER(Uc_v, Uc_p);
    VECTOR_VIEW_CLOSE_POINTER(out_v, out_p);
    MViewTime += usecond();
    MTotalTime += usecond();
    grid_printf_flush("Finished calling: DhopDirInternal_gpu\n");
  }

  void DhopDirInternal_cpu(Stencil& stencil, const PhysicalGaugeField& Uc, const FermionField& in, uint64_t in_n_virtual, FermionField& out, uint64_t out_n_virtual, VirtualFermionField& tmp, int point) {
    DhopDirInternal_gpu(stencil, Uc, in, in_n_virtual, out, out_n_virtual, tmp, point);
  }

  void MooeeInternal_gpu(const PhysicalLinkField& UcSelf, const FermionField& in, uint64_t in_n_virtual, FermionField& out, uint64_t out_n_virtual) {
    MCalls++;
    MTotalTime -= usecond();
    MMiscTime -= usecond();
    const int Narg            = numArg(in, in_n_virtual, out, out_n_virtual);
    const int NvirtualFermion = in_n_virtual;
    const int NvirtualLink    = NvirtualFermion*NvirtualFermion;
    const int Nsimd           = SiteSpinor::Nsimd();
    const int Nsite           = in[0].Grid()->oSites();

    assert(n_arg_ == Narg);

    constantCheckerboard(in, out);

    MViewTime -= usecond();
    VECTOR_VIEW_OPEN_POINTER(UcSelf, UcSelf_v, UcSelf_p, AcceleratorRead);
    VECTOR_VIEW_OPEN_POINTER(in, in_v, in_p, AcceleratorRead);
    VECTOR_VIEW_OPEN_POINTER(out, out_v, out_p, AcceleratorWrite);
    MViewTime += usecond();

    for(int v_col=0; v_col<NvirtualFermion; v_col++) {
      typedef decltype(coalescedRead(in_v[0][0]))    calcVector;
      typedef decltype(coalescedRead(in_v[0][0](0))) calcComplex;

      MComputeTime -= usecond();
      accelerator_for(idx, Nsite*NvirtualFermion*Narg*NbasisVirtual, Nsimd, {
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
          res = coalescedRead(out_p[v_arg_row][ss](b));
#endif

        nbr = coalescedRead(in_v[v_arg_col][ss]);

        for(int bb=0;bb<NbasisVirtual;bb++) {
          res = res + coalescedRead(UcSelf_p[v_link][ss](b,bb))*nbr(bb);
        }
#if defined(REFERENCE_SUMMATION_ORDER)
        if (v_col != 0) {
          res = res + coalescedRead(out_p[v_arg_row][ss](b));
        }
#endif
        coalescedWrite(out_p[v_arg_row][ss](b),res);
      });
      MComputeTime += usecond();
    }
    MViewTime -= usecond();
    VECTOR_VIEW_CLOSE_POINTER(UcSelf_v, UcSelf_p);
    VECTOR_VIEW_CLOSE_POINTER(in_v, in_p);
    VECTOR_VIEW_CLOSE_POINTER(out_v, out_p);
    MViewTime += usecond();
    MTotalTime += usecond();
    grid_printf_flush("Finished calling: MooeeInternal_gpu\n");
  }

  void MooeeInternal_cpu(const PhysicalLinkField& UcSelf, const FermionField& in, uint64_t in_n_virtual, FermionField& out, uint64_t out_n_virtual) {
    MooeeInternal_gpu(UcSelf, in, in_n_virtual, out, out_n_virtual);
  }

  void MooeeDagInternal_gpu(const PhysicalLinkField& UcSelf, const FermionField& in, uint64_t in_n_virtual, FermionField& out, uint64_t out_n_virtual) {
    MCalls++;
    MTotalTime -= usecond();
    MMiscTime -= usecond();
    const int Narg            = numArg(in, in_n_virtual, out, out_n_virtual);
    const int NvirtualFermion = in_n_virtual;
    const int NvirtualLink    = NvirtualFermion*NvirtualFermion;
    const int Nsimd           = SiteSpinor::Nsimd();
    const int Nsite           = in[0].Grid()->oSites();

    assert(n_arg_ == Narg);

    constantCheckerboard(in, out);

    MViewTime -= usecond();
    VECTOR_VIEW_OPEN_POINTER(UcSelf, UcSelf_v, UcSelf_p, AcceleratorRead);
    VECTOR_VIEW_OPEN_POINTER(in, in_v, in_p, AcceleratorRead);
    VECTOR_VIEW_OPEN_POINTER(out, out_v, out_p, AcceleratorWrite);
    RealD* dag_factor_p = &dag_factor_[0];
    MViewTime += usecond();

    for(int v_col=0; v_col<NvirtualFermion; v_col++) {
      typedef decltype(coalescedRead(in_v[0][0]))    calcVector;
      typedef decltype(coalescedRead(in_v[0][0](0))) calcComplex;

      MComputeTime -= usecond();
      accelerator_for(idx, Nsite*NvirtualFermion*Narg*NbasisVirtual, Nsimd, {
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
          res = coalescedRead(out_p[v_arg_row][ss](b));
#endif

        nbr = coalescedRead(in_v[v_arg_col][ss]);

        for(int bb=0;bb<NbasisVirtual;bb++) {
          const int bb_global = v_row*NvirtualFermion+bb;
          res = res + dag_factor_p[b_global*nbasis_global_+bb_global] * coalescedRead(UcSelf_p[v_link][ss](b,bb))*nbr(bb);
        }
#if defined(REFERENCE_SUMMATION_ORDER)
        if (v_col != 0) {
          res = res + coalescedRead(out_p[v_arg_row][ss](b));
        }
#endif
        coalescedWrite(out_p[v_arg_row][ss](b),res);
      });
      MComputeTime += usecond();
    }
    MViewTime -= usecond();
    VECTOR_VIEW_CLOSE_POINTER(UcSelf_v, UcSelf_p);
    VECTOR_VIEW_CLOSE_POINTER(in_v, in_p);
    VECTOR_VIEW_CLOSE_POINTER(out_v, out_p);
    MViewTime += usecond();
    MTotalTime += usecond();
    grid_printf_flush("Finished calling: MooeeDagInternal_gpu\n");
  }

  void MooeeDagInternal_cpu(const PhysicalLinkField& UcSelf, const FermionField& in, uint64_t in_n_virtual, FermionField& out, uint64_t out_n_virtual) {
    MooeeDagInternal_gpu(UcSelf, in, in_n_virtual, out, out_n_virtual);
  }
};
