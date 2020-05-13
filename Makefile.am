
# libtools preferrs this
ACLOCAL_AMFLAGS = -I m4

SUBDIRS = lib

# make the cgpt a libtool library.
# Conveniently, this will take care of the rest (prefix, suffix,
# linking, fPIC, ....)

pyexec_LTLIBRARIES = cgpt.la

cgpt_la_SOURCES = \
	lib/cgpt/lib/operators.cc \
	lib/cgpt/lib/init.cc \
	lib/cgpt/lib/eval.cc \
	lib/cgpt/lib/block.cc \
	lib/cgpt/lib/coordinates.cc \
	lib/cgpt/lib/expression_gammamul.cc \
	lib/cgpt/lib/mpi.cc \
	lib/cgpt/lib/lattice.cc \
	lib/cgpt/lib/transform.cc \
	lib/cgpt/lib/expression_linear_combination.cc \
	lib/cgpt/lib/io.cc \
	lib/cgpt/lib/expression_mul.cc \
	lib/cgpt/lib/util.cc \
	lib/cgpt/lib/random.cc \
	lib/cgpt/lib/grid.cc \
	lib/cgpt/lib/distribute.cc \
	lib/cgpt/lib/basis.cc \
	lib/cgpt/lib/time.cc \
	lib/cgpt/lib/lib.cc

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
cgpt_la_CPPFLAGS = $(PYTHON_CPPFLAGS) $(GRID_CPPFLAGS)
cgpt_la_LIBADD = $(GRID_LDFLAGS) $(GRID_LIBS)

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
