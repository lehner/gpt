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
void lattice_init();

// type name shortcuts
#define BASIS_SIZE(n) \
  template<typename vtype> using iVSinglet ## n = iVector<iSinglet<vtype>,n>; \
  template<typename vtype> using iMSinglet ## n = iMatrix<iSinglet<vtype>,n>;
#include "../basis_size.h"
#undef BASIS_SIZE

#define SPIN_COLOR(Ns,Nc)						\
  template<typename vtype> using iMSpin ## Ns ## Color ## Nc = iScalar<iMatrix<iMatrix<vtype, Nc>, Ns> >; \
  template<typename vtype> using iVSpin ## Ns ## Color ## Nc = iScalar<iVector<iVector<vtype, Nc>, Ns> >;

#define SPIN(Ns)							\
  template<typename vtype> using iMSpin ## Ns = iScalar<iMatrix<iScalar<vtype>, Ns> >; \
  template<typename vtype> using iVSpin ## Ns = iScalar<iVector<iScalar<vtype>, Ns> >;

#define COLOR(Nc)							\
  template<typename vtype> using iMColor ## Nc = iScalar<iScalar<iMatrix<vtype, Nc> > > ; \
  template<typename vtype> using iVColor ## Nc = iScalar<iScalar<iVector<vtype, Nc> > > ;

#include "../spin_color.h"

#undef SPIN_COLOR
#undef SPIN
#undef COLOR

// map precision to string
static const std::string get_prec(const vComplexF& l) { return "single"; };
static const std::string get_prec(const vComplexD& l) { return "double"; };
template<typename T> const std::string get_prec(const Lattice<T>& l) { typedef typename Lattice<T>::vector_type vCoeff_t; vCoeff_t t; return get_prec(t); }

// map objects to their type string
template<typename vtype> const std::string get_otype(const iSinglet<vtype>& l) { return "ot_singlet"; };
template<typename vtype, int Nc> const std::string get_otype(const iScalar<iScalar<iMatrix<vtype, Nc> > >& l) { return "ot_mcolor" + std::to_string(Nc); };
template<typename vtype, int Nc> const std::string get_otype(const iScalar<iScalar<iVector<vtype, Nc> > >& l) { return "ot_vcolor" + std::to_string(Nc); };
template<typename vtype, int Ns> const std::string get_otype(const iScalar<iMatrix<iScalar<vtype>, Ns> >& l) { return "ot_mspin" + std::to_string(Ns); };
template<typename vtype, int Ns> const std::string get_otype(const iScalar<iVector<iScalar<vtype>, Ns> >& l) { return "ot_vspin" + std::to_string(Ns); };
template<typename vtype, int Ns, int Nc> const std::string get_otype(const iScalar<iMatrix<iMatrix<vtype, Nc>, Ns> >& l) { return "ot_mspin" + std::to_string(Ns) + "color" + std::to_string(Nc); };
template<typename vtype, int Ns, int Nc> const std::string get_otype(const iScalar<iVector<iVector<vtype, Nc>, Ns> >& l) { return "ot_vspin" + std::to_string(Ns) + "color" + std::to_string(Nc); };
template<typename vtype,int nbasis> const std::string get_otype(const iVector<iSinglet<vtype>,nbasis>& l) { return std::string("ot_vsinglet") + std::to_string(nbasis); };
template<typename vtype,int nbasis> const std::string get_otype(const iMatrix<iSinglet<vtype>,nbasis>& l) { return std::string("ot_msinglet") + std::to_string(nbasis); };
template<typename T> const std::string get_otype(const Lattice<T>& l) { typedef typename Lattice<T>::vector_object vobj; vobj t; return get_otype(t); }

// map objects to their singlet rank (0 = scalar, 1 = vector, 2 = matrix)
template<typename vtype> int singlet_rank(const iSinglet<vtype>& l) { return 0; };
template<typename vtype, int Nc> int singlet_rank(const iScalar<iScalar<iMatrix<vtype, Nc> > >& l) { return 0; };
template<typename vtype, int Nc> int singlet_rank(const iScalar<iScalar<iVector<vtype, Nc> > >& l) { return 0; };
template<typename vtype, int Ns> int singlet_rank(const iScalar<iMatrix<iScalar<vtype>, Ns> >& l) { return 0; };
template<typename vtype, int Ns> int singlet_rank(const iScalar<iVector<iScalar<vtype>, Ns> >& l) { return 0; };
template<typename vtype, int Ns, int Nc> int singlet_rank(const iScalar<iMatrix<iMatrix<vtype, Nc>, Ns> >& l) { return 0; };
template<typename vtype, int Ns, int Nc> int singlet_rank(const iScalar<iVector<iVector<vtype, Nc>, Ns> >& l) { return 0; };
template<typename vtype,int nbasis> int singlet_rank(const iVector<iSinglet<vtype>,nbasis>& l) { return 1; };
template<typename vtype,int nbasis> int singlet_rank(const iMatrix<iSinglet<vtype>,nbasis>& l) { return 2; };
template<typename T> int singlet_rank(const Lattice<T>& l) { typedef typename Lattice<T>::vector_object vobj; vobj t; return singlet_rank(t); }

// map otype strings to their singlet rank
extern std::map<std::string,int> _otype_singlet_rank_;

// singlet rank and linear size to dimension
static
int size_to_singlet_dim(int rank, int size) {
  if (rank == 0) {
    ASSERT(size == 1);
    return 1;
  } else if (rank == 1) {
    return size;
  } else if (rank == 2) {
    int sqrt_size = (int)(sqrt(size) + 0.5);
    ASSERT(sqrt_size * sqrt_size == size);
    return sqrt_size;
  } else {
    ERR("Unknown singlet_rank %d",rank);
  }
}
