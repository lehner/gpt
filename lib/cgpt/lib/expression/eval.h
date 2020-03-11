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

cgpt_Lattice_base* eval_general(cgpt_Lattice_base* dst, std::vector<_eval_term_>& terms,int unary) {

  // first turn every factor
  std::vector< cgpt_lattice_term > lterms;
  for (size_t i=0;i<terms.size();i++) {
    auto& term = terms[i];
    ASSERT(term.factors.size() > 0);
    if (term.factors.size() == 1) {
      auto& factor = term.factors[0];
      ASSERT(factor.type == _eval_factor_::LATTICE);
      lterms.push_back( cgpt_lattice_term( term.coefficient, factor.lattice, false ) );
    } else {
      // calculate factor
      ASSERT(0);
      //      cgpt_Lattice_base* c_second = a.second->mul( b.second, a.first, b.first, 0 );
    }
  }

  ASSERT(lterms.size() > 0);

  dst = lterms[0].get_lat()->compatible_linear_combination(dst,false,lterms, unary);

  for (auto& term : lterms)
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

  std::vector<_eval_term_> terms;
  eval_convert_factors(_list,terms);

  // Three fast cases
  // 1) a*A*B + C
  // 2) a*A + b*B + c*C + ...
  // 3) A*B*C*D*...

  dst=eval_general(dst,terms,_unary);

  if (dst)
    return dst->to_decl();

  return PyLong_FromLong(0);

} EXPORT_END();
