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
#include "lib.h"

// declare
#define EXPORT_FUNCTION(name) extern PyObject* cgpt_ ## name(PyObject* self, PyObject* args);
#include "exports.h"
#undef EXPORT_FUNCTION

// add to module functions
#define EXPORT_FUNCTION(name) {# name, cgpt_ ## name, METH_VARARGS, # name},
static PyMethodDef module_functions[] = {
#include "exports.h"
  {NULL, NULL, 0, NULL}
};
#undef EXPORT_FUNCTION

// on exit
void free_module(void* self) {
  cgpt_exit((PyObject*)self,0);
}

// module definition
static struct PyModuleDef module_def = {
  PyModuleDef_HEAD_INIT,
  "cgpt",     /* m_name */
  "The C++ interface between gpt and Grid",  /* m_doc */
  -1,                  /* m_size */
  module_functions,    /* m_methods */
  NULL,                /* m_reload */
  NULL,                /* m_traverse */
  NULL,                /* m_clear */
  free_module,         /* m_free */
};

// export module creation
PyMODINIT_FUNC PyInit_cgpt(void){
  lattice_init();
  return PyModule_Create(&module_def);
}
