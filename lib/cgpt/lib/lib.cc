/*
  CGPT

  Authors: Christoph Lehner 2020
*/
#include "lib.h"

#define EXPORT_FUNCTION(name) {# name, cgpt_ ## name, METH_VARARGS, # name}
#define DECLARE_FUNCTION(name) extern PyObject* cgpt_ ## name(PyObject* self, PyObject* args);

DECLARE_FUNCTION(init);
DECLARE_FUNCTION(create_grid);
DECLARE_FUNCTION(delete_grid);
DECLARE_FUNCTION(grid_barrier);
DECLARE_FUNCTION(grid_globalsum);
DECLARE_FUNCTION(create_lattice);
DECLARE_FUNCTION(delete_lattice);
DECLARE_FUNCTION(lattice_set_val);
DECLARE_FUNCTION(lattice_to_str);
DECLARE_FUNCTION(lattice_axpy_norm);
DECLARE_FUNCTION(lattice_norm2);
DECLARE_FUNCTION(lattice_innerProduct);
DECLARE_FUNCTION(lattice_sum);
DECLARE_FUNCTION(cshift);
DECLARE_FUNCTION(copy);
DECLARE_FUNCTION(eval);
DECLARE_FUNCTION(global_rank);
DECLARE_FUNCTION(load);


static PyMethodDef module_functions[] = {
  EXPORT_FUNCTION(init),
  EXPORT_FUNCTION(create_grid),
  EXPORT_FUNCTION(delete_grid),
  EXPORT_FUNCTION(grid_barrier),
  EXPORT_FUNCTION(grid_globalsum),
  EXPORT_FUNCTION(create_lattice),
  EXPORT_FUNCTION(delete_lattice),
  EXPORT_FUNCTION(lattice_set_val),
  EXPORT_FUNCTION(lattice_to_str),
  EXPORT_FUNCTION(lattice_axpy_norm),
  EXPORT_FUNCTION(lattice_norm2),
  EXPORT_FUNCTION(lattice_innerProduct),
  EXPORT_FUNCTION(lattice_sum),
  EXPORT_FUNCTION(cshift),
  EXPORT_FUNCTION(copy),
  EXPORT_FUNCTION(eval),
  EXPORT_FUNCTION(global_rank),
  EXPORT_FUNCTION(load),
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
