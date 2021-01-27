/*
    GPT - Grid Python Toolkit
    Copyright (C) 2020  Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)
                  2020  Daniel Richtmann (daniel.richtmann@ur.de, https://github.com/lehner/gpt)

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

template<typename T>
class eigenConverter {
public:
  
  eigenConverter(long n_virtual) {
    ERR("Not implemented");
  }

  Eigen::MatrixXcd matrix() {
    return Eigen::MatrixXcd::Zero(0,0);
  }

  template<typename sobj>
    void fillMatrix(long lex_outer, Eigen::MatrixXcd & matrix, const sobj & obj) {
  }
  
  template<typename sobj>
    void fillObject(long lex_outer, sobj & obj, const Eigen::MatrixXcd & matrix) {
  }
};

template<typename CComplex, int basis_virtual_size>
class eigenConverter<iMatrix<iSinglet<CComplex>, basis_virtual_size>> {
public:
  
  long n_virtual_red;
  long nbasis_global;

  eigenConverter(long n_virtual) {
    n_virtual_red = long(sqrt(n_virtual));
    nbasis_global = n_virtual_red * basis_virtual_size;
  }

  Eigen::MatrixXcd matrix() {
    return Eigen::MatrixXcd::Zero(nbasis_global, nbasis_global);
  }

  template<typename sobj>
  void fillMatrix(long lex_outer, Eigen::MatrixXcd & matrix, const sobj & obj) {
    // convention for indices:
    // lex_outer = lexicographic index in array of v_objs
    // row_outer = row index    in array of v_objs (column-major ordering)
    // col_outer = column index in array of v_objs (column-major ordering)
    // row_inner = row index    inside a v_obj = grid matrix tensor (row-major ordering)
    // col_inner = column index inside a v_obj = grid matrix tensor (row-major ordering)
    // row_global = row index    of combination of v_objs viewed as 1 single big matrix tensor (row-major ordering)
    // col_global = column index of combination of v_objs viewed as 1 single big matrix tensor (row-major ordering)

    long row_outer = lex_outer % n_virtual_red;
    long col_outer = lex_outer / n_virtual_red;
    for (long row_inner=0; row_inner<basis_virtual_size; row_inner++) {
      for (long col_inner=0; col_inner<basis_virtual_size; col_inner++) {
	long row_global = row_outer * basis_virtual_size + row_inner;
	long col_global = col_outer * basis_virtual_size + col_inner;
	matrix(row_global, col_global) = static_cast<ComplexD>(TensorRemove(obj(row_inner, col_inner)));
      }
    }
  }

  template<typename sobj>
  void fillObject(long lex_outer, sobj & obj, const Eigen::MatrixXcd & matrix) {
    long row_outer = lex_outer % n_virtual_red;
    long col_outer = lex_outer / n_virtual_red;
    for (long row_inner=0; row_inner<basis_virtual_size; row_inner++) {
      for (long col_inner=0; col_inner<basis_virtual_size; col_inner++) {
	long row_global = row_outer * basis_virtual_size + row_inner;
	long col_global = col_outer * basis_virtual_size + col_inner;
	obj(row_inner, col_inner) = matrix(row_global, col_global);
      }
    }
  }
};


template<typename stype, int Ncolor>
class eigenConverter<iScalar<iScalar<iMatrix<stype, Ncolor>>>> {
public:
  
  eigenConverter(long n_virtual) {
    ASSERT(n_virtual == 1);
  }

  Eigen::MatrixXcd matrix() {
    return Eigen::MatrixXcd::Zero(Ncolor, Ncolor);
  }

  template<typename sobj>
    void fillMatrix(long lex_outer, Eigen::MatrixXcd & matrix, const sobj & obj) {
    for(long row=0;row<Ncolor;row++)
      for(long col=0;col<Ncolor;col++)
	matrix(row, col) = static_cast<ComplexD>(TensorRemove(obj()()(row, col)));
  }

  template<typename sobj>
    void fillObject(long lex_outer, sobj & obj, const Eigen::MatrixXcd & matrix) {
    for(long row=0;row<Ncolor;row++)
      for(long col=0;col<Ncolor;col++)
	obj()()(row, col) = matrix(row, col);
  }
};

template<typename stype, int Nspin>
  class eigenConverter<iScalar<iMatrix<iScalar<stype>,Nspin>>> {
public:
  
  eigenConverter(long n_virtual) {
    ASSERT(n_virtual == 1);
  }

  Eigen::MatrixXcd matrix() {
    return Eigen::MatrixXcd::Zero(Nspin, Nspin);
  }

  template<typename sobj>
    void fillMatrix(long lex_outer, Eigen::MatrixXcd & matrix, const sobj & obj) {
    for(long row=0;row<Nspin;row++)
      for(long col=0;col<Nspin;col++)
	matrix(row, col) = static_cast<ComplexD>(TensorRemove(obj()(row, col)()));
  }
  
  template<typename sobj>
    void fillObject(long lex_outer, sobj & obj, const Eigen::MatrixXcd & matrix) {
    for(long row=0;row<Nspin;row++)
      for(long col=0;col<Nspin;col++)
	obj()(row, col)() = matrix(row, col);
  }
};

template<typename stype, int Nspin, int Ncolor>
  class eigenConverter<iScalar<iMatrix<iMatrix<stype,Ncolor>,Nspin>>> {
public:
  
  eigenConverter(long n_virtual) {
    ASSERT(n_virtual == 1);
  }

  Eigen::MatrixXcd matrix() {
    return Eigen::MatrixXcd::Zero(Nspin*Ncolor, Nspin*Ncolor);
  }

  template<typename sobj>
    void fillMatrix(long lex_outer, Eigen::MatrixXcd & matrix, const sobj & obj) {
    for (long s_row=0;s_row<Nspin;s_row++)
      for (long s_col=0;s_col<Nspin;s_col++)
	for (long c_row=0;c_row<Ncolor;c_row++)
	  for (long c_col=0;c_col<Ncolor;c_col++)
	    matrix(s_row*Ncolor+c_row, s_col*Ncolor+c_col) = static_cast<ComplexD>(TensorRemove(obj()(s_row, s_col)(c_row, c_col)));
    
  }
  
  template<typename sobj>
    void fillObject(long lex_outer, sobj & obj, const Eigen::MatrixXcd & matrix) {
    for (long s_row=0;s_row<Nspin;s_row++)
      for (long s_col=0;s_col<Nspin;s_col++)
	for (long c_row=0;c_row<Ncolor;c_row++)
	  for (long c_col=0;c_col<Ncolor;c_col++)
	    obj()(s_row, s_col)(c_row, c_col) = matrix(s_row*Ncolor+c_row, s_col*Ncolor+c_col);
  }
};

