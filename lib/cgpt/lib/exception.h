/*
  CGPT

  Authors: Christoph Lehner 2020
*/
#define ASSERT(x)				\
  { if ( !(x) )throw "Assert " #x " failed"; };

#define EXPORT_BEGIN(name)					   \
  static PyObject* cgpt_ ## name(PyObject* self, PyObject* args) { \
    try {								   

#define EXPORT_END()						   \
    } catch (const char* err) {					   \
      PyErr_SetString(PyExc_RuntimeError,err);			   \
      return NULL;						   \
    }								   \
  }
