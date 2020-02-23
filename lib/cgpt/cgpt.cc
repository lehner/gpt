#include <Python.h>
#include <vector>
#include <string>
#include <iostream>

#include <Grid/Grid.h>

using namespace Grid;

#include "lib/convert.h"
#include "lib/grid.h"
#include "lib/peekpoke.h"
#include "lib/lattice.h"
#include "lib/mul.h"
#include "lib/transform.h"
#include "lib/init.h"
#include "lib/mpi.h"

static PyMethodDef module_functions[] = {
  {"init", cgpt_init, METH_VARARGS, "Initializes gpt"},
  {"create_grid", cgpt_create_grid, METH_VARARGS, "Creates a grid"},
  {"delete_grid", cgpt_delete_grid, METH_VARARGS, "Deletes a grid"},
  {"grid_barrier", cgpt_grid_barrier, METH_VARARGS, "Grid::Barrier"},
  {"grid_globalsum", cgpt_grid_globalsum, METH_VARARGS, "Grid global sum"},
  {"create_lattice", cgpt_create_lattice, METH_VARARGS, "Creates a lattice"},
  {"delete_lattice", cgpt_delete_lattice, METH_VARARGS, "Deletes a lattice"},
  {"lattice_set_val", cgpt_lattice_set_val, METH_VARARGS, "Set a value within a lattice"},
  {"lattice_to_str", cgpt_lattice_to_str, METH_VARARGS, "Get a string representation of the lattice"},
  {"lattice_mul", cgpt_lattice_mul, METH_VARARGS, "Multiply two lattices"},
  {"lattice_axpy_norm", cgpt_lattice_axpy_norm, METH_VARARGS, "axpy_norm"},
  {"lattice_adj", cgpt_lattice_adj, METH_VARARGS, "Adjungate"},
  {"lattice_norm2", cgpt_lattice_norm2, METH_VARARGS, "Norm2"},
  {"lattice_innerProduct", cgpt_lattice_innerProduct, METH_VARARGS, "innerProduct"},
  {"cshift", cgpt_cshift, METH_VARARGS, "Cshift"},
  {"copy", cgpt_copy, METH_VARARGS, "Copy"},
  {"eval", cgpt_eval, METH_VARARGS, "Evaluate linear combinations"},
  {"global_rank", cgpt_global_rank, METH_VARARGS, "Rank within global MPI"},
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
