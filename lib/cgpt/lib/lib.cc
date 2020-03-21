/*
  CGPT

  Authors: Christoph Lehner 2020
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
  PyModule_Create(&module_def);
}
