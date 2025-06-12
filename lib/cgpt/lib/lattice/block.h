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
class cgpt_block_map_base {
public:
  virtual ~cgpt_block_map_base() { };

  virtual void project(std::vector<cgpt_Lattice_base*>& _coarse, long coarse_n_virtual,
		       std::vector<cgpt_Lattice_base*>& _fine, long fine_n_virtual) = 0;
  virtual void promote(std::vector<cgpt_Lattice_base*>& _coarse, long coarse_n_virtual,
		       std::vector<cgpt_Lattice_base*>& _fine, long fine_n_virtual)  = 0;
  virtual void orthonormalize() = 0;
  virtual void sum(std::vector<cgpt_Lattice_base*>& _coarse, long coarse_n_virtual,
		   std::vector<cgpt_Lattice_base*>& _fine, long fine_n_virtual) = 0;
  virtual void embed(std::vector<cgpt_Lattice_base*>& _coarse, long coarse_n_virtual,
		     std::vector<cgpt_Lattice_base*>& _fine, long fine_n_virtual) = 0;
};


template<typename obj>
struct cgpt_project_identity {
  typedef typename obj::scalar_object tensor_t;

  void* get_args() { return 0; }

  cgpt_project_identity(PyObject* args) { }
  
  static accelerator_inline obj projector(obj x, long idx, void* args) {
    return x;
  }

  static accelerator_inline tensor_t projector(tensor_t x, long idx, void* args) {
    return x;
  }

};

template<typename obj>
struct cgpt_project_tensor {
  typedef typename obj::scalar_object tensor_t;
  typedef typename obj::scalar_type scalar_t;
  typedef typename obj::vector_type vector_t;
  
  Vector<tensor_t> tensors;

  void* get_args() {
    return (void*)&tensors[0];
  }

  void cgpt_convert(PyObject* src, tensor_t& dst) {
    cgpt_numpy_import(dst, src);
  }
  
  cgpt_project_tensor(PyObject* args) {
    ASSERT(PyList_Check(args));
    
    tensors.resize(PyList_Size(args));
    for (size_t i = 0; i < tensors.size(); i++)
      cgpt_convert(PyList_GetItem(args,i),tensors[i]);
  }

  static accelerator_inline obj projector(obj x, long idx, void* args) {
    tensor_t* p_tensor = (tensor_t*)args;

    vector_t* s = (vector_t*)&x;
    scalar_t* t = (scalar_t*)&p_tensor[idx];
    for (int i=0;i<sizeof(tensor_t) / sizeof(scalar_t);i++)
      s[i] *= t[i];
    return x;
  }

  static accelerator_inline tensor_t projector(tensor_t x, long idx, void* args) {
    tensor_t* p_tensor = (tensor_t*)args;

    scalar_t* s = (scalar_t*)&x;
    scalar_t* t = (scalar_t*)&p_tensor[idx];
    for (int i=0;i<sizeof(tensor_t) / sizeof(scalar_t);i++)
      s[i] *= t[i];
    return x;
  }
};

template<class T, class C, typename projector_t>
class cgpt_block_map : public cgpt_block_map_base {

  typedef typename Lattice<T>::vector_object vobj;
  typedef typename vobj::scalar_object sobj;
  typedef typename Lattice<T>::vector_type vCoeff_t;
  typedef typename Lattice<T>::scalar_type Coeff_t;
  typedef iSinglet<vCoeff_t> T_singlet;

  PVector<Lattice<T>> basis;
  long basis_n_virtual, basis_n, basis_n_block;
  cgpt_block_lookup_table<T_singlet> lut;
  GridBase* coarse_grid;
  GridBase* fine_grid;

  projector_t projector;
  
public:

  virtual ~cgpt_block_map() {
  }

  cgpt_block_map(GridBase* _coarse_grid, 
		 std::vector<cgpt_Lattice_base*>& _basis, long _basis_n_virtual, long _basis_n_block,
		 cgpt_Lattice_base* _mask, PyObject* tensor_projectors)
    :
    lut(_coarse_grid, compatible<T_singlet>(_mask)->l),
    coarse_grid(_coarse_grid),
    fine_grid(_basis[0]->get_grid()),
    basis_n_virtual(_basis_n_virtual),
    basis_n_block(_basis_n_block),
    projector(tensor_projectors)
  {
    cgpt_basis_fill(basis,_basis);
    ASSERT(basis.size() % basis_n_virtual == 0);
    basis_n = basis.size() / basis_n_virtual;
  }

  virtual void project(std::vector<cgpt_Lattice_base*>& _coarse, long coarse_n_virtual,
		       std::vector<cgpt_Lattice_base*>& _fine, long fine_n_virtual) {

    PVector<Lattice<T>> fine;
    PVector<Lattice<C>> coarse;
    cgpt_basis_fill(fine,_fine);
    cgpt_basis_fill(coarse,_coarse);

    vectorizableBlockProject(coarse, coarse_n_virtual, fine, fine_n_virtual, basis, basis_n_virtual, lut, basis_n_block, projector);
  }

  virtual void promote(std::vector<cgpt_Lattice_base*>& _coarse, long coarse_n_virtual,
		       std::vector<cgpt_Lattice_base*>& _fine, long fine_n_virtual) {


    PVector<Lattice<T>> fine;
    PVector<Lattice<C>> coarse;
    cgpt_basis_fill(fine,_fine);
    cgpt_basis_fill(coarse,_coarse);

    vectorizableBlockPromote(coarse, coarse_n_virtual, fine, fine_n_virtual, basis, basis_n_virtual, lut, basis_n_block, projector);
  }

  virtual void orthonormalize() {
    Lattice<T_singlet> c(coarse_grid);
    vectorBlockOrthonormalize(c,basis,basis_n_virtual);
  }

  virtual void sum(std::vector<cgpt_Lattice_base*>& _coarse, long coarse_n_virtual,
		   std::vector<cgpt_Lattice_base*>& _fine, long fine_n_virtual) {

    PVector<Lattice<T>> fine;
    PVector<Lattice<T>> coarse;
    cgpt_basis_fill(fine,_fine);
    cgpt_basis_fill(coarse,_coarse);

    vectorizableBlockSum(coarse, coarse_n_virtual, fine, fine_n_virtual, lut);
  }

  virtual void embed(std::vector<cgpt_Lattice_base*>& _coarse, long coarse_n_virtual,
		     std::vector<cgpt_Lattice_base*>& _fine, long fine_n_virtual) {

    PVector<Lattice<T>> fine;
    PVector<Lattice<T>> coarse;
    cgpt_basis_fill(fine,_fine);
    cgpt_basis_fill(coarse,_coarse);

    vectorizableBlockEmbed(coarse, coarse_n_virtual, fine, fine_n_virtual, lut);
  }
};
