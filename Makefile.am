
# libtools preferrs this
ACLOCAL_AMFLAGS = -I m4

SUBDIRS = lib

# make the cgpt a libtool library.
# Conveniently, this will take care of the rest (prefix, suffix,
# linking, fPIC, ....)

pyexec_LTLIBRARIES = cgpt.la

include $(top_srcdir)/cgpt_ccfiles.inc
include $(top_srcdir)/gpt_pyfiles.inc

cgpt_la_SOURCES = $(CGPT_CCFILES)

# The flags in ..._LDFLAGS are __not__ linker flags, but flags for
# GNU libtool. It takes care of all the platform specific stuff when
# creating shared library objects:
# -avoid-version   avoid the linux typicall *.so.major.minor stuff.
# -module          avoid putting lib* as prefix.
# -shared          make it a shared object
# -export-dynamic  Very important in our case and easily overlooked:
#                  cgpt itself will link to another library. But
#                  when loaded by python, don't use @rpath or other
#                  tricks, but python will dlopen the necessary symbols.
#                  To avoit errors, use this flag!
cgpt_la_LDFLAGS = -avoid-version -module -export-dynamic -shared

# Compiler Flags.
# ..._CFLAGS not needed, since this is a pure C++ library.
# ..._CXXFLAGS  is set for compiling and linking,
# ..._CPPFLAGS  is set only for compiling (not for linking)
# ..._LIBADD    are flags for linking
# cgpt_la_CFLAGS = $(PYTHON_CFLAGS) $(GRID_CFLAGS)
cgpt_la_CXXFLAGS = $(PYTHON_CFLAGS) $(GRID_CXXFLAGS)
cgpt_la_CPPFLAGS = $(PYTHON_CPPFLAGS) $(GRID_CPPFLAGS) \
		   -I$(GRID_PREFIX)/include \
		   -I$(NUMPY_INCLUDE)
cgpt_la_LIBADD = -L$(GRID_PREFIX)/lib -lGrid \
		 $(GRID_LDFLAGS) $(GRID_LIBS)

# link dependencies go into <programname>_LDADD
# libtool figures out all the -l  and -L compiler flags for you
# if this has an *.la  file

# Additional flags shoudl go into AM_CPPFLAGS.
# There is a big difference between setting CFLAGS or AM_CFLAGS
# If the user has CFLAGS defined e.g., as a shell environment variable,
# the value defined here in Makefile.am will be overwritten!
# The values of AM_CFLAGS will be prepended to CFLAGS.
# Therefore, default values should usually go into AM_CFLAGS

# to make a tarball, try "make dist"
# and to check for errors/forgotten files, use "make distcheck".
#
