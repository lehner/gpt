/*
  CGPT

  Authors: Christoph Lehner 2020

  Description:  We need to fail gracefully since we also run in an interpreter; infrastructure goes here
*/
#define STRX(x) #x
#define STR(x) STRX(x)
#define ASSERT(x)				\
  { if ( !(x) )throw "Assert " #x " failed in file " __FILE__ ":"  STR(__LINE__); };
#define ERR(...)							\
  { char msg[1024]; snprintf(msg,sizeof(msg)-100,__VA_ARGS__);		\
    strcat(msg, " in file " __FILE__ ":"  STR(__LINE__)); throw (const char*)msg; };

#define EXPORT(name,...)					   \
  PyObject* cgpt_ ## name(PyObject* self, PyObject* args) {	   \
    try {							   \
      __VA_ARGS__;						   \
    } catch (const char* err) {					   \
      PyErr_SetString(PyExc_RuntimeError,err);			   \
      return NULL;						   \
    }								   \
  }
