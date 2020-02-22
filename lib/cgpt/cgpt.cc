#include <Python.h>
#include <vector>
#include <string>
#include <iostream>

#include "convert.h"
#include "delete.h"
#include "grid.h"

static PyMethodDef module_functions[] = {
  {"create_grid", cgpt_create_grid, METH_VARARGS, "Creates a grid"},
  {"delete", cgpt_delete, METH_VARARGS, "Deletes an object"},
  {NULL, NULL, 0, NULL}
};

// Module definition
static struct PyModuleDef module_def = {
  PyModuleDef_HEAD_INIT,
  "cgpt",     /* m_name */
  "The C++ interface between gpt and Grid",  /* m_doc */
  -1,                  /* m_size */
  module_functions,    /* m_methods */
  NULL,                /* m_reload */
  NULL,                /* m_traverse */
  NULL,                /* m_clear */
  NULL,                /* m_free */
};

// Export module creation
PyMODINIT_FUNC PyInit_cgpt(void){
  PyModule_Create(&module_def);
}
