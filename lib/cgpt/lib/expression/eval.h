/*
  CGPT

  Authors: Christoph Lehner 2020
*/
struct _eval_factor_ {
  enum { LATTICE, GAMMA } type;
  int unary;
  union {
    cgpt_Lattice_base* lattice;
    int gamma;
  };

  void release() {
    if (type == LATTICE)
      delete lattice;
  }
};

struct _eval_term_ {
  ComplexD coefficient;
  std::vector<_eval_factor_> factors;
};

void eval_convert_factors(PyObject* _list, std::vector<_eval_term_>& terms) {
  ASSERT(PyList_Check(_list));
  int n = (int)PyList_Size(_list);

  terms.resize(n);
  for (int i=0;i<n;i++) {
    auto& term = terms[i];
    PyObject* tp = PyList_GetItem(_list,i);
    ASSERT(PyTuple_Check(tp) && PyTuple_Size(tp) == 2);

    cgpt_convert(PyTuple_GetItem(tp,0),term.coefficient);

    PyObject* ll = PyTuple_GetItem(tp,1);
    ASSERT(PyList_Check(ll));
    int m = (int)PyList_Size(ll);

    term.factors.resize(m);
    for (int j=0;j<m;j++) {
      auto& factor = term.factors[j];
      PyObject* jj = PyList_GetItem(ll,j);
      ASSERT(PyTuple_Check(jj) && PyTuple_Size(jj) == 2);

      factor.unary = PyLong_AsLong(PyTuple_GetItem(jj,0));
      PyObject* f = PyTuple_GetItem(jj,1);
      if (PyObject_HasAttrString(f,"obj")) {
	factor.lattice = (cgpt_Lattice_base*)PyLong_AsVoidPtr(PyObject_GetAttrString(f,"obj"));
	factor.type = _eval_factor_::LATTICE;
      } else {
	ASSERT(0);
      }
    }
  }
}

_eval_factor_ eval_mul_factor(_eval_factor_ lhs, _eval_factor_ rhs, int unary) {

  _eval_factor_ dst;
  dst.unary = 0;

  if (lhs.type == _eval_factor_::LATTICE) {
    if (rhs.type == _eval_factor_::LATTICE) {
      dst.type = _eval_factor_::LATTICE;
      dst.lattice = lhs.lattice->mul( 0, false, rhs.lattice, lhs.unary, rhs.unary, unary);
    } else {
      ASSERT(0);
    }
  } else {
    ASSERT(0);
  }

  return dst;
}

cgpt_Lattice_base* eval_term(std::vector<_eval_factor_>& factors, int term_unary) {
  ASSERT(factors.size() > 1);

  //f[im2] = eval_mul_factor(f[im2],f[im1]);
  //f[im3] = eval_mul_factor(f[im3],f[im2]);
  //f[im2].release();
  //f[im4] = eval_mul_factor(f[im4],f[im3]);
  //f[im3].release();

  for (size_t i = factors.size() - 1; i > 0; i--) {
    auto& im1 = factors[i];
    auto& im2 = factors[i-1];
    im2 = eval_mul_factor(im2,im1, i == 1 ? term_unary : 0); // apply unary operator in last step
    if (i != factors.size() - 1)
      im1.release();
  }

  ASSERT(factors[0].type == _eval_factor_::LATTICE);
  return factors[0].lattice;
}

cgpt_Lattice_base* eval_general(cgpt_Lattice_base* dst, std::vector<_eval_term_>& terms,int unary) {

  // class A)
  // first separate all terms that are a pure linear combination:
  //   result_class_a = unary(A + B + C) taken using compatible_linear_combination

  // class B)
  // for all other terms, create terms and apply unary operators before summing
  //   result_class_b = unary(B*C*D) + unary(E*F)

  std::vector< cgpt_lattice_term > terms_a, terms_b;

  for (size_t i=0;i<terms.size();i++) {
    auto& term = terms[i];
    ASSERT(term.factors.size() > 0);
    if (term.factors.size() == 1) {
      auto& factor = term.factors[0];
      ASSERT(factor.type == _eval_factor_::LATTICE);
      terms_a.push_back( cgpt_lattice_term( term.coefficient, factor.lattice, false ) );
    } else {
      terms_b.push_back( cgpt_lattice_term( term.coefficient, eval_term(term.factors, unary), true ) );
    }
  }

  if (terms_a.size() > 0) {
    dst = terms_a[0].get_lat()->compatible_linear_combination(dst,false, terms_a, unary);
    if (terms_b.size() > 0) {
      dst = terms_b[0].get_lat()->compatible_linear_combination(dst,true, terms_b, 0); // unary operators have been applied above
    }
  } else {
    if (terms_b.size() > 0) {
      dst = terms_b[0].get_lat()->compatible_linear_combination(dst,false, terms_b, 0); // unary operators have been applied above
    } else {
      ERR("no terms to be added!");
    }
  }

  for (auto& term : terms_b)
    term.release();

  return dst;
}

EXPORT_BEGIN(eval) {

  void* _dst;
  PyObject* _list;
  int _unary;
  if (!PyArg_ParseTuple(args, "lOi", &_dst, &_list, &_unary)) {
    return NULL;
  }

  cgpt_Lattice_base* dst = (cgpt_Lattice_base*)_dst;
  bool new_lattice = dst == 0;

  std::vector<_eval_term_> terms;
  eval_convert_factors(_list,terms);

  dst=eval_general(dst,terms,_unary);

  if (new_lattice)
    return dst->to_decl();

  return PyLong_FromLong(0);

} EXPORT_END();
