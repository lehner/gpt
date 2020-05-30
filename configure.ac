#                                               -*- Autoconf -*-

# You can specifiy a minimal autoconf version here, but you don't have to
AC_PREREQ([2.69])

# project name, version and maintainer's address
AC_INIT([gpt],[0.1],[stefan.solbrig@ur.de])

# separate directory for m4 stuff (libtools prefers this)
AC_CONFIG_MACRO_DIR([m4])

# you have to list one file that does already exist. Can be any file.
AC_CONFIG_SRCDIR([lib/gpt/__init__.py])

# use configure header
AC_CONFIG_HEADERS([gpt_config.h])

# Trick to set empty default value for CXXFLAGS
: ${CXXFLAGS=""}


# archiver (Automake needs this for libraries)
AM_PROG_AR

# need for dynamic libraries (no need for ranlib)
LT_INIT

# check for Python
# If you change them minimal Python version, do it in line line
# below (m4 syntax!) in order to change it consistently for the
# interpreter and the header files.
AC_DEFUN([my_python_version],[3.6])
AM_PATH_PYTHON(my_python_version)
AX_PYTHON_DEVEL([>=']my_python_version['])

AC_ARG_WITH(grid,
    AS_HELP_STRING(
        [--with-grid=DIR],
        [Specify grid installation directory DIR]
  ),
  [GRID_HOME="$with_grid"]
)

AC_ARG_ENABLE([fake-grid],
              [AS_HELP_STRING([--enable-fake-grid],
                              [Fake grid-config. Useful if you just want
                               to do a 'make dist'])],
              [ac_fake_grid=yes],
              [ac_fake_grid=no])

# Numpy include dir
AC_DEFUN([numpy_includer],["${PYTHON}" -c 'import numpy; print(numpy.get_include())'])
AC_SUBST([NUMPY_INCLUDE])
AC_MSG_CHECKING([if we can find numpy headers])
AS_IF([numpy_includer > /dev/null 2>&1 ],
      [NUMPY_INCLUDE="$(numpy_includer)"
       AC_MSG_RESULT([${NUMPY_INCLUDE}])],
      [AC_MSG_ERROR([no.])])

AS_IF([test "x${ac_fake_grid}" = "xyes"],
[
    AC_SUBST([GRID_CXXFLAGS],[-DFAKEGRID])
    AC_SUBST([GRID_LDFLAGS],[-LFAKEGRID])
    AC_SUBST([GRID_CXX],[gcc])
    AC_SUBST([GRID_CXXLD],[gcc])
    AC_SUBST([GRID_LIBS],  [-lFAKEGRID])
    AC_SUBST([GRID_PREFIX],[FAKEGRID])
],
[
AS_IF([test "x${GRID_HOME}x" = "xx"],
    [AC_PATH_PROG(GRID_CONFIG, [grid-config], [])],
    [AC_PATH_PROG(GRID_CONFIG, [grid-config], [], [${GRID_HOME}/bin:${PATH}])])
AS_IF([test "x${GRID_CONFIG}x" = "xx"],
      [AC_MSG_ERROR([cannot find grid-config])],
      [])
AC_SUBST([GRID_CXXFLAGS],[`${GRID_CONFIG} --cxxflags`])
AC_SUBST([GRID_LDFLAGS],[`${GRID_CONFIG} --ldflags`])
AC_SUBST([GRID_CXX],[`${GRID_CONFIG} --cxx`])
AC_SUBST([GRID_CXXLD],[`${GRID_CONFIG} --cxxld`])
AC_SUBST([GRID_LIBS],  [`${GRID_CONFIG} --libs`])
AC_SUBST([GRID_PREFIX],[`${GRID_CONFIG} --prefix`])
])

# Unfortunately, not all versions of Grid have --cxxld and it does
# not print to stderr in case of an error. So detect error by
# matching the output to Usage:

AS_IF([expr "${GRID_CXXLD}" : ".*Usage:" > /dev/null],
      [GRID_CXXLD=])

# If GRID_CXXLD is unset or null, use GRID_CXX.
# Set CXX to GRID_CXX if CXX is unset or null. This means, by default
# use Grid's CXX, but user can override it.  Same for GRID_CXXLD.

: ${GRID_CXXLD:=${GRID_CXX}}
: ${CXX:=${GRID_CXX}}
: ${CXXLD:=${GRID_CXXLD}}

# C++ -Compiler
AC_PROG_CXX

# The following code tries to fix nvcc (and possibly other weird
# compilers) that take an -Xcompiler or -Xlinker argument flag.
# These flags are also used by GNU libtool, hence a conflict arises.
# Fix the conflict by 'escaping' the problematic flags.

AC_PROG_SED
AC_DEFUN([fixxer],[# The FiXXer
AS_IF([expr "${$1}" : ".*$2 " > /dev/null],
      [$1=`echo "${$1}" | ${SED} -e "s/$2 /$2 $2 $2 /g"`])])

for token in -Xlinker -Xcompiler; do
fixxer([GRID_CXXFLAGS],${token})
done

# Use automake
# the next macro takes a space separated list of arguments.
# First argument is the minimal automake version. foreign means that
# you don't follow GNU standards (ChangeLog COPYING files etc.)
# and -Wall turns on all automake warnings.
AM_INIT_AUTOMAKE([1.16.1 foreign subdir-objects -Wall])

# Makefiles to be generated. These are all *.in  files.
# We have to use subdirs, because for the python files I need to
# use nobase_XYZ.
AC_CONFIG_FILES([Makefile lib/Makefile])

# Generate output. This has to be the last command!
AC_OUTPUT
