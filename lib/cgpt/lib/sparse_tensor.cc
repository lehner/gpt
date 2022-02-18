/*
    GPT - Grid Python Toolkit
    Copyright (C) 2022  Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)

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
#include "lib.h"
#include "sparse_tensor.h"

EXPORT(create_tensor_basis,{
    
    PyObject* _basis;
    if (!PyArg_ParseTuple(args, "O", &_basis)) {
      return NULL;
    }

    std::shared_ptr<tensor_basis>* r = new std::shared_ptr<tensor_basis>(new tensor_basis());

    std::vector<std::vector<long>> basis;
    cgpt_convert(_basis,basis);

    for (auto & b : basis) {
      ASSERT(b.size() == 2);

      (*r)->dimensions.push_back({b[0],b[1]});

    }
    
    return PyLong_FromVoidPtr(r);
  });

EXPORT(delete_tensor_basis,{
    
    void* p;
    if (!PyArg_ParseTuple(args, "l", &p)) {
      return NULL;
    }

    std::shared_ptr<tensor_basis>* b = (std::shared_ptr<tensor_basis>*)p;
    delete b;
    
    return PyLong_FromLong(0);
  });

EXPORT(tensor_basis_get,{
    
    void* p;
    PyObject* a;
    if (!PyArg_ParseTuple(args, "lO", &p, &a)) {
      return NULL;
    }

    std::shared_ptr<tensor_basis>* b = (std::shared_ptr<tensor_basis>*)p;
    
    if (a == Py_None) {
      return PyLong_FromLong((*b)->dimensions.size());
    } else {
      ASSERT(PyLong_Check(a));
      long l = PyLong_AsLong(a);
      ASSERT(l >= 0 && l < (*b)->dimensions.size());

      PyObject* r = PyTuple_New(2);
      PyTuple_SET_ITEM(r, 0, PyLong_FromLong((*b)->dimensions[l].first));
      PyTuple_SET_ITEM(r, 1, PyLong_FromLong((*b)->dimensions[l].second));
      return r;
    }
  });

EXPORT(create_sparse_tensor,{
    
    void* p;
    long n;
    if (!PyArg_ParseTuple(args, "ll", &p,&n)) {
      return NULL;
    }

    std::shared_ptr<tensor_basis>* b = (std::shared_ptr<tensor_basis>*)p;

    return PyLong_FromVoidPtr(new std::vector<sparse_tensor>(n, *b));
  });

EXPORT(delete_sparse_tensor,{
    
    void* p;
    if (!PyArg_ParseTuple(args, "l", &p)) {
      return NULL;
    }

    std::vector<sparse_tensor>* t = (std::vector<sparse_tensor>*)p;
    delete t;
    
    return PyLong_FromLong(0);
  });

EXPORT(sparse_tensor_set,{
    
    void* p;
    PyObject* lst;
    if (!PyArg_ParseTuple(args, "lO", &p, &lst)) {
      return NULL;
    }

    std::vector<sparse_tensor>* t = (std::vector<sparse_tensor>*)p;
    long n_parallel = t->size();

    ASSERT(PyList_Check(lst));
    ASSERT(PyList_Size(lst) == n_parallel);

    for (long tid=0;tid<n_parallel;tid++) {

      PyObject* d = PyList_GetItem(lst, tid);
      
      ASSERT(PyDict_Check(d));
      
      PyObject *_key, *_value;
      Py_ssize_t pos = 0;
      
      while (PyDict_Next(d, &pos, &_key, &_value)) {
        ComplexD value;
        cgpt_convert(_value, value);
        
        ASSERT(PyTuple_Check(_key));
        
        int n = (*t)[tid].basis->dimensions.size();
        ASSERT(PyTuple_Size(_key) == n);
        
        std::vector<int> c(n);
        for (int i=0;i<n;i++) {
          PyObject* ki = PyTuple_GET_ITEM(_key, i);
          ASSERT(PyLong_Check(ki));
          c[i] = PyLong_AsLong(ki);
        }
        
        (*t)[tid].values[(*t)[tid].basis->c2i(c)] = value;
      }

    }
    
    return PyLong_FromLong(0);
  });

EXPORT(sparse_tensor_get,{
    
    void* p;
    PyObject* _key;
    if (!PyArg_ParseTuple(args, "lO", &p, &_key)) {
      return NULL;
    }

    std::vector<sparse_tensor>* t = (std::vector<sparse_tensor>*)p;

    long n_parallel = t->size();

    PyObject* lst = PyList_New(n_parallel);
    
    if (_key == Py_None) {

      for (long tid=0;tid<n_parallel;tid++) {

        PyObject* r = PyDict_New();
        
        int n = (*t)[tid].basis->dimensions.size();
        for (auto & v : (*t)[tid].values) {
          
          auto c = (*t)[tid].basis->i2c(v.first);
          PyObject* k = PyTuple_New(n);
          for (int i=0;i<n;i++) {
            PyTuple_SET_ITEM(k, i, PyLong_FromLong(c[i]));
          }
          
          PyObject* val = PyComplex_FromDoubles(v.second.real(), v.second.imag());
          
          PyDict_SetItem(r, k, val);
          
          Py_XDECREF(k);
          Py_XDECREF(val);
          
        }

        PyList_SetItem(lst, tid, r);     
      }

    } else {
      ASSERT(PyTuple_Check(_key));

      for (long tid=0;tid<n_parallel;tid++) {
        
        int n = (*t)[tid].basis->dimensions.size();
        ASSERT(PyTuple_Size(_key) == n);
      
        std::vector<int> c(n);
        for (int i=0;i<n;i++) {
          PyObject* ki = PyTuple_GET_ITEM(_key, i);
          ASSERT(PyLong_Check(ki));
          c[i] = PyLong_AsLong(ki);
        }
      
        auto f = (*t)[tid].values.find((*t)[tid].basis->c2i(c));
        if (f == (*t)[tid].values.end()) {
          PyList_SetItem(lst, tid, Py_None);
        } else {
          PyList_SetItem(lst, tid, PyComplex_FromDoubles(f->second.real(), f->second.imag()));
        }
      }

    }

    return lst;
  });


EXPORT(sparse_tensor_binary,{
    
    void* p1, * p2;
    long l;
    PyObject* _p2;
    if (!PyArg_ParseTuple(args, "lOl", &p1, &_p2, &l)) {
      return NULL;
    }

    std::vector<sparse_tensor>* t1 = (std::vector<sparse_tensor>*)p1;
    long n_parallel = t1->size();
    std::vector<sparse_tensor>* r = new std::vector<sparse_tensor>(n_parallel, (*t1)[0].basis);


    if (PyLong_Check(_p2)) {

      p2 = (void*)PyLong_AsLong(_p2);

      std::vector<sparse_tensor>* t2 = (std::vector<sparse_tensor>*)p2;
      ASSERT(n_parallel == t2->size());
    
      if (l) {
        thread_for(i, n_parallel, {
            (*r)[i] = (*t1)[i] * (*t2)[i];
          });
      } else {
        thread_for(i, n_parallel, {
            (*r)[i] = (*t1)[i] + (*t2)[i];
          });
      }

    } else {

      ASSERT(PyComplex_Check(_p2));
      ComplexD val = ComplexD(PyComplex_RealAsDouble(_p2), PyComplex_ImagAsDouble(_p2));

      thread_for(i, n_parallel, {
          (*r)[i] = (*t1)[i] * val;
        });

    }

    PyObject* ret = PyTuple_New(2);
    PyTuple_SetItem(ret, 0, PyLong_FromVoidPtr(r));
    PyTuple_SetItem(ret, 1, PyLong_FromVoidPtr(new std::shared_ptr<tensor_basis>((*r)[0].basis)));
    return ret;
    
  });


EXPORT(sparse_tensor_contract,{
    
    PyObject* _tensors, * _symbols;
    if (!PyArg_ParseTuple(args, "OO", &_tensors, &_symbols)) {
      return NULL;
    }

    ASSERT(PyList_Check(_tensors));
    ASSERT(PyList_Check(_symbols));
    
    long ntensors = PyList_Size(_tensors);
    long nsymbols = PyList_Size(_symbols);
    ASSERT(ntensors > 0);
    long n_parallel;

    std::vector<std::vector<sparse_tensor>*> tensors(ntensors);
    for (long it = 0;it<ntensors;it++) {
      PyObject* x = PyList_GetItem(_tensors, it);
      ASSERT(PyLong_Check(x));
      tensors[it] = (std::vector<sparse_tensor>*)PyLong_AsLong(x);
      if (it == 0)
        n_parallel = tensors[it]->size();
      else
        ASSERT(n_parallel == tensors[it]->size());
    }

    std::set<int> symbols;
    for (long it=0;it<nsymbols;it++) {
      PyObject* x = PyList_GetItem(_symbols, it);
      ASSERT(PyLong_Check(x));
      symbols.insert( (int)PyLong_AsLong(x) );
    }

    std::vector<sparse_tensor>* r = new std::vector<sparse_tensor>(n_parallel, (*tensors[0])[0].basis);

    thread_for(i, n_parallel, {
        std::vector<sparse_tensor*> ctensors(ntensors);
        for (int j=0;j<ntensors;j++)
          ctensors[j] = &(*tensors[j])[i];
        (*r)[i] = sparse_tensor::contract(ctensors, symbols);
      });

    PyObject* ret = PyTuple_New(2);
    PyTuple_SetItem(ret, 0, PyLong_FromVoidPtr(r));
    PyTuple_SetItem(ret, 1, PyLong_FromVoidPtr(new std::shared_ptr<tensor_basis>((*r)[0].basis)));
    return ret;
  });


EXPORT(sparse_tensor_sum,{
    
    void* p;
    if (!PyArg_ParseTuple(args, "l", &p)) {
      return NULL;
    }

    std::vector<sparse_tensor>* t = (std::vector<sparse_tensor>*)p;
    long n_parallel = t->size();
    std::vector<sparse_tensor>* r = new std::vector<sparse_tensor>(1, (*t)[0].basis);

    for (long i=0;i<n_parallel;i++)
      (*r)[0] = (*r)[0] + (*t)[i];
  
    PyObject* ret = PyTuple_New(2);
    PyTuple_SetItem(ret, 0, PyLong_FromVoidPtr(r));
    PyTuple_SetItem(ret, 1, PyLong_FromVoidPtr(new std::shared_ptr<tensor_basis>((*r)[0].basis)));
    return ret;
    
  });
